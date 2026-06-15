import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os

# 在 base.py 顶部添加
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory, self._snr_memory = np.array([]), np.array([]), np.array([])
        self.topk = 3

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        #self._device = args["device"][0]
        self._device = torch.device("cuda" if args["device"] and torch.cuda.is_available() else "cpu")
        self._multiple_gpus = args["device"]


        self.snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.acc_pred_snr = torch.zeros(len(self.snrs))
        self.acc_total_snr = torch.zeros(len(self.snrs))
        self.acc_snr = torch.zeros(len(self.snrs))

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class // self._total_classes
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes



    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        # if self._fixed_memory:
        #     self._construct_exemplar_unified(data_manager, per_class)
        # else:
        #     self._reduce_exemplar(data_manager, per_class)
        #     self._construct_exemplar(data_manager, per_class)
        self._reduce_exemplar(data_manager, per_class)
        self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def \
            eval_task(self, save_conf=False):

        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory, self._snr_memory)
    #
    # def _compute_accuracy(self, model, loader):
    #     model.eval()
    #     correct, total = 0, 0
    #
    #
    #
    #     for i, (_, inputs1, targets, snr) in enumerate(loader):
    #         inputs = inputs1.unsqueeze(-1).repeat(1, 1, 1, 128)
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             outputs = model(inputs)["logits"]
    #         predicts = torch.max(outputs, dim=1)[1]
    #         correct += (predicts.cpu() == targets).sum()
    #         total += len(targets)
    #
    #         for ik in range(targets.shape[0]):
    #             idx = self.snrs.index(snr[ik])
    #             self.acc_pred_snr[idx] += (targets[ik] == predicts[ik]).cpu()  # 记录每个信噪比下的正确预测样本数，初始化为全零。
    #             self.acc_total_snr[idx] += 1  # 记录每个信噪比下的总样本数，初始化为全零
    #     for j in range(len(self.snrs)):
    #         self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]
    #
    #
    #     return np.around(tensor2numpy(correct) * 100 / total, decimals=2), self.acc_snr

    # def _compute_accuracy(self, model, loader):
    #     model.eval()
    #     correct, total = 0, 0
    #
    #     # === 新增代码：初始化分组统计 ===
    #     group_ranges = {
    #         'low': (0, 4),
    #         'mid': (5, 7),
    #         'high': (8, 10)
    #     }
    #     group_correct = {k: 0 for k in group_ranges}
    #     group_total = {k: 0 for k in group_ranges}
    #     # =============================
    #
    #     for i, (_, inputs1, targets, snr) in enumerate(loader):
    #         inputs = inputs1.unsqueeze(-1).repeat(1, 1, 1, 128)
    #         inputs = inputs.to(self._device)
    #         with torch.no_grad():
    #             outputs = model(inputs)["logits"]
    #         predicts = torch.max(outputs, dim=1)[1]
    #         correct += (predicts.cpu() == targets).sum()
    #         total += len(targets)
    #
    #         for ik in range(targets.shape[0]):
    #             idx = self.snrs.index(snr[ik])
    #             self.acc_pred_snr[idx] += (targets[ik] == predicts[ik]).cpu()
    #             self.acc_total_snr[idx] += 1
    #
    #             # === 新增代码：更新分组统计 ===
    #             target_label = targets[ik].item()
    #             predict_label = predicts[ik].item()
    #
    #             for group_name, (start, end) in group_ranges.items():
    #                 if start <= target_label <= end:
    #                     group_total[group_name] += 1
    #                     if target_label == predict_label:
    #                         group_correct[group_name] += 1
    #                     break  # 找到对应分组后跳出循环
    #             # =============================
    #
    #     for j in range(len(self.snrs)):
    #         self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]
    #
    #     # === 新增代码：计算分组准确率 ===
    #     group_acc = {}
    #     for group in group_ranges:
    #         acc = group_correct[group] / group_total[group] if group_total[group] > 0 else 0.0
    #         group_acc[group] = np.around(acc * 100, 2)
    #     # =============================
    #
    #     # 保持原有返回结构，新增分组结果作为第三个返回值
    #     return (np.around(tensor2numpy(correct) * 100 / total, decimals=2),
    #         self.acc_snr,
    #         group_acc  # 新增的第三个返回值
    #             )

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0

        self.acc_pred_snr = torch.zeros(len(self.snrs))
        self.acc_total_snr = torch.zeros(len(self.snrs))
        self.acc_snr = torch.zeros(len(self.snrs))
        # ===== 新增代码：初始化分组SNR统计 =====
        group_ranges = [(0, 4), (5, 7), (8, 10)]  # 定义分组边界
        # 每个SNR维护3个分组的统计量 [low, mid, high]
        group_pred_snr = [[0] * 3 for _ in range(len(self.snrs))]  # 正确数
        group_total_snr = [[0] * 3 for _ in range(len(self.snrs))]  # 总数
        # ====================================

        for i, (_, inputs, targets, snr) in enumerate(loader):
            # inputs = inputs1.unsqueeze(-1).repeat(1, 1, 1, 128)
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

            for ik in range(targets.shape[0]):
                # 原有SNR统计逻辑
                idx = self.snrs.index(snr[ik])
                self.acc_pred_snr[idx] += (targets[ik] == predicts[ik]).cpu()
                self.acc_total_snr[idx] += 1

                # ===== 新增代码：分组SNR统计 =====
                target_label = targets[ik].item()
                # 确定分组索引 (0:low, 1:mid, 2:high)
                group_idx = next(
                    (i for i, (s, e) in enumerate(group_ranges) if s <= target_label <= e),
                    -1
                )
                if group_idx != -1 and idx < len(self.snrs):
                    group_total_snr[idx][group_idx] += 1
                    if targets[ik] == predicts[ik]:
                        group_pred_snr[idx][group_idx] += 1
                # ================================

        # 原有SNR准确率计算
        for j in range(len(self.snrs)):
            self.acc_snr[j] = self.acc_pred_snr[j] / self.acc_total_snr[j]

        # ===== 新增代码：计算分组SNR准确率 =====
        snr_group_acc = {}
        for snr_idx, snr_val in enumerate(self.snrs):
            acc_dict = {}
            for group_idx, (s, e) in enumerate(group_ranges):
                total1 = group_total_snr[snr_idx][group_idx]
                correct1 = group_pred_snr[snr_idx][group_idx]
                acc = correct1 / total1 if total1 > 0 else 0.0
                acc_dict[f"{s}-{e}"] = np.around(acc * 100, 2)
            snr_group_acc[snr_val] = acc_dict
        # ===================================

        # 保持原有返回结构，新增第三个返回值
        return (
            np.around(tensor2numpy(correct) * 100 / total, decimals=2),
            self.acc_snr,
            snr_group_acc  # 新增的SNR分组准确率
        )
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets, _) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets, _ in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets, dummy_snr = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        ), copy.deepcopy(self._snr_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory, self._snr_memory = np.array([]), np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt, ds = dummy_data[mask][:m], dummy_targets[mask][:m], dummy_snr[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )
            self._snr_memory = (
                np.concatenate((self._snr_memory, ds))
                if len(self._snr_memory) != 0
                else ds
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt, ds)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, snrs, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            selected_exemplars_snr = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                selected_exemplars_snr.append(snrs[i])
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            exemplar_snr = np.array(selected_exemplars_snr)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            self._snr_memory = (
                np.concatenate((self._snr_memory, exemplar_snr))
                if len(self._snr_memory) != 0
                else exemplar_snr
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets, exemplar_snr),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets, class_snr = (
                self._data_memory[mask],
                self._targets_memory[mask],
                self._snr_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets, class_snr)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, snrs, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            selected_exemplars_snr = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                selected_exemplars_snr.append(snrs[i])
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            exemplar_snr = np.array(selected_exemplars_snr)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            self._snr_memory = (
                np.concatenate((self._snr_memory, exemplar_snr))
                if len(self._snr_memory) != 0
                else exemplar_snr
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets, exemplar_snr),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

        # ... 把这段代码加在 BaseLearner 类里面 ...

    # def plot_tsne(self, loader, task_id, save_name=None, max_samples_per_class=200):
    #     """
    #     执行 t-SNE 并保存图片
    #     :param loader: 数据加载器
    #     :param task_id: 当前任务ID
    #     :param save_name: 保存的文件名，如果为None则自动生成
    #     :param max_samples_per_class: 为了绘图清晰，每类最多随机选取的样本数
    #     """
    #     print(f"Extraction features for t-SNE (Task {task_id})...")
    #     self._network.eval()
    #
    #     # 1. 提取所有特征和标签
    #     vectors, targets = self._extract_vectors(loader)
    #
    #     # 2. 采样 (避免点太多看不清，也为了加速)
    #     # 将 numpy 转为 DataFrame 方便采样
    #     df_full = pd.DataFrame(vectors)
    #     df_full['label'] = targets
    #
    #     # 每类随机采样
    #     df_sampled = df_full.groupby('label').apply(
    #         lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=42))
    #     # 去掉索引
    #     df_sampled = df_sampled.reset_index(drop=True)
    #
    #     sampled_vectors = df_sampled.iloc[:, :-1].values
    #     sampled_targets = df_sampled['label'].values
    #
    #     print(f"Running t-SNE on {len(sampled_vectors)} samples...")
    #
    #     # 3. 运行 t-SNE
    #     # init='pca' 通常能得到更稳定的结果
    #     tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    #     X_embedded = tsne.fit_transform(sampled_vectors)
    #
    #     # 4. 绘图
    #     plt.figure(figsize=(10, 10))
    #     # 使用 seaborn 绘制，不同类别不同颜色
    #     # palette 可以改，比如 'viridis', 'deep', 'tab10'
    #     sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1],
    #                     hue=sampled_targets, palette='tab10',
    #                     legend='full', s=60, alpha=0.8)
    #
    #     plt.title(f"Feature t-SNE Task {task_id} ({self.args['model_name']})")
    #     plt.xlabel("Dimension 1")
    #     plt.ylabel("Dimension 2")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #
    #     # 5. 保存
    #     if save_name is None:
    #         save_dir = os.path.join(f"logs/{self.args['model_name']}/{self.args['dataset']}/")
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         save_path = os.path.join(save_dir, f"tsne_task_{task_id}_1.png")
    #     else:
    #         save_path = save_name
    #
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()
    #     print(f"t-SNE plot saved to {save_path}")

    # ---------------------------------------------------------
    # 新增功能：提取带 SNR 的特征
    # ---------------------------------------------------------
    def _extract_vectors_with_snr(self, loader):
        self._network.eval()
        vectors, targets, snrs = [], [], []
        # 注意：这里假设 loader 返回 (index, inputs, targets, snr)
        # 根据 icarl.py 的逻辑，loader 确实返回这四个值
        for _, _inputs, _targets, _snr in loader:
            _targets = _targets.numpy()
            _snr = _snr.numpy()

            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)
            snrs.append(_snr)

        return np.concatenate(vectors), np.concatenate(targets), np.concatenate(snrs)

    # ---------------------------------------------------------
    # 新增功能：绘制特定类别在不同 SNR 下的 t-SNE
    # ---------------------------------------------------------
    def plot_tsne_snr_specific_class(self, loader, task_id, target_class_idx, class_name="Class",
                                     save_prefix="baseline"):
        """
        绘制特定类别在不同 SNR 下的 t-SNE 分布图
        :param loader: 数据加载器
        :param task_id: 当前任务 ID
        :param target_class_idx: 目标类别的索引 (int)，例如 QPSK 对应的数字索引
        :param class_name: 目标类别的名称 (str)，例如 'QPSK'，用于图表标题
        :param save_prefix: 保存文件的前缀，通常用于区分 'baseline' 或 'ours'
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import os

        print(f"Extraction features for SNR t-SNE (Task {task_id}, Class {class_name})...")

        # 1. 提取特征、标签和 SNR
        vectors, targets, snrs = self._extract_vectors_with_snr(loader)

        # 2. 筛选特定类别的数据
        mask = (targets == target_class_idx)
        class_vectors = vectors[mask]
        class_snrs = snrs[mask]

        print(f"Total samples for {class_name}: {len(class_vectors)}")

        if len(class_vectors) > 1000:
            # 随机采样 1000 个点，避免绘图过慢且过于拥挤
            indices = np.random.choice(len(class_vectors), 1000, replace=False)
            class_vectors = class_vectors[indices]
            class_snrs = class_snrs[indices]

        # 3. 运行 t-SNE
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        X_embedded = tsne.fit_transform(class_vectors)

        # 4. 绘图
        plt.figure(figsize=(10, 8))

        # 使用 SNR 作为颜色映射 (Hue)
        # palette 选用 'coolwarm' 或 'viridis' 能很好地展示 SNR 从低到高的渐变
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1],
                        hue=class_snrs, palette='viridis',
                        legend='full', s=80, alpha=0.9, edgecolor='w')

        plt.title(f"t-SNE for {class_name} across SNRs\nModel: {save_prefix} | Task: {task_id}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        # 调整图例，显示 SNR
        plt.legend(title='SNR (dB)', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 5. 保存
        save_dir = os.path.join(f"logs/tsne_results/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = f"{save_prefix}_task{task_id}_{class_name}_snr_tsne.png"
        save_path = os.path.join(save_dir, filename)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"SNR t-SNE plot saved to {save_path}")
