import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

init_epoch = 50
init_lr = 0.1
init_milestones = [50]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 50
lrate = 0.1
milestones = [50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8


class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        self.snrs = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.acc_pred_snr = torch.zeros(len(self.snrs))
        self.acc_total_snr = torch.zeros(len(self.snrs))
        self.acc_snr = torch.zeros(len(self.snrs))
        self.ref_signals = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        self.acc_pred_snr = torch.zeros(len(self.snrs))
        self.acc_total_snr = torch.zeros(len(self.snrs))
        self.acc_snr = torch.zeros(len(self.snrs))

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        A, B = [], []
        self.last_5_epochs_acc = []
        self.snr_to_idx = {snr: idx for idx, snr in enumerate(self.snrs)}

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            current_acc_pred_snr = [0] * len(self.snrs)
            current_acc_total_snr = [0] * len(self.snrs)

            for i, (idx, inputs, targets, snr) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                for idx_sample in range(targets.shape[0]):
                    snr_val = snr[idx_sample].item()
                    if snr_val not in self.snr_to_idx:
                        continue
                    idx = self.snr_to_idx[snr_val]
                    current_acc_pred_snr[idx] += (targets[idx_sample] == preds[idx_sample]).cpu().item()
                    current_acc_total_snr[idx] += 1

            # 计算当前epoch每个SNR的准确率
            current_epoch_acc = []
            for j in range(len(self.snrs)):
                acc = current_acc_pred_snr[j] / current_acc_total_snr[j] if current_acc_total_snr[j] > 0 else 0.0
                current_epoch_acc.append(acc)

            # 保存当前epoch结果到列表，只保留最后5个
            self.last_5_epochs_acc.append(current_epoch_acc)
            if len(self.last_5_epochs_acc) > 5:
                self.last_5_epochs_acc.pop(0)  # 移除最早的epoch结果

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            A.extend([train_acc])

            if epoch % 2 == 0:
                test_acc, test_acc_snr, group_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    loss,
                    train_acc,
                    test_acc,
                )
                B.append([test_acc])
                # //
                print("Test_accy = ", test_acc)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    loss,
                    train_acc,
                )

            prog_bar.set_description(info)

        # 训练全部结束后，计算最后5个epoch的平均值
        self.avg_last_5_acc = []
        for j in range(len(self.snrs)):
            # 对每个SNR，取最后5个epoch的准确率平均值
            sum_acc = sum(epoch_acc[j] for epoch_acc in self.last_5_epochs_acc)
            avg_acc = sum_acc / len(self.last_5_epochs_acc)
            self.avg_last_5_acc.append(avg_acc)

        print(A)
        print(B)
        print(self.avg_last_5_acc)
        print(test_acc_snr)
        print(group_acc)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))

        A, B = [], []
        self.last_5_epochs_acc = []
        self.snr_to_idx = {snr: idx for idx, snr in enumerate(self.snrs)}

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            current_acc_pred_snr = [0] * len(self.snrs)
            current_acc_total_snr = [0] * len(self.snrs)

            for i, (idx, inputs, targets, snr) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets.long()
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                for idx_sample in range(targets.shape[0]):
                    snr_val = snr[idx_sample].item()
                    if snr_val not in self.snr_to_idx:
                        continue
                    idx = self.snr_to_idx[snr_val]
                    current_acc_pred_snr[idx] += (targets[idx_sample] == preds[idx_sample]).cpu().item()
                    current_acc_total_snr[idx] += 1

            # 计算当前epoch每个SNR的准确率
            current_epoch_acc = []
            for j in range(len(self.snrs)):
                acc = current_acc_pred_snr[j] / current_acc_total_snr[j] if current_acc_total_snr[j] > 0 else 0.0
                current_epoch_acc.append(acc)

            # 保存当前epoch结果到列表，只保留最后5个
            self.last_5_epochs_acc.append(current_epoch_acc)
            if len(self.last_5_epochs_acc) > 5:
                self.last_5_epochs_acc.pop(0)  # 移除最早的epoch结果


            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            A.extend([train_acc])

            if epoch % 2 == 0:
                test_acc, test_acc_snr, group_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    loss,
                    train_acc,
                    test_acc,
                )
                B.append([test_acc])
                print("Test_accy = ", test_acc)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    loss,
                    train_acc,
                )
            prog_bar.set_description(info)

        self.avg_last_5_acc = []
        for j in range(len(self.snrs)):
            # 对每个SNR，取最后5个epoch的准确率平均值
            sum_acc = sum(epoch_acc[j] for epoch_acc in self.last_5_epochs_acc)
            avg_acc = sum_acc / len(self.last_5_epochs_acc)
            self.avg_last_5_acc.append(avg_acc)

        print(A)
        print(B)
        print(self.avg_last_5_acc)
        print(test_acc_snr)
        print(group_acc)

        logging.info(info)
