import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000
from tqdm import tqdm
import pickle
import torch
import random


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self.Random_Matrix = None
        self.transformed_clean_data = None

        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[:task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y, snr = self._train_data, self._train_targets, self._train_snr
        elif source == "test":
            x, y, snr = self._test_data, self._test_targets, self._test_snr
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            # //
            # trsf = transforms.Compose([
            #     transforms.ToTensor(),
            #     None,
            # ])
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        # //
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets, snrs = [], [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets, class_snr = self._select(
                    x, y, snr, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets, class_snr = self._select_rmm(
                    x, y, snr, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
            snrs.append(class_snr)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets, appendent_snr = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            snrs.append(appendent_snr)

        data, targets, snrs = np.concatenate(data), np.concatenate(targets), np.concatenate(snrs)

        # # 新增：获取干净信号（同类型且SNR=18）
        # # ------------------------------
        # if self.transformed_clean_data is None:
        #     clean_data = []
        #     # 预构建类别到SNR=18信号的映射（提高查找效率）
        #     class_clean_signal_map = {}
        #
        #     for class_idx in np.unique(targets):
        #         # 筛选该类别下SNR=18的信号
        #         mask = (y == class_idx) & (np.isclose(snr, 18.0))
        #         if np.sum(mask) == 0:
        #             raise ValueError(f"类别 {class_idx} 没有SNR=18的信号，请检查数据集")
        #         class_clean_signal_map[class_idx] = x[mask]
        #
        #     # 为每个样本匹配同类别干净信号
        #     for i in range(len(targets)):
        #         class_idx = targets[i]
        #         # 从该类别干净信号中随机选择一个（可根据需求改为固定选择）
        #         clean_sig = random.choice(class_clean_signal_map[class_idx])
        #         clean_data.append(clean_sig)
        #
        #     clean_data = np.stack(clean_data, axis=0)  # 转换为数组格式
        #     # 应用数据变换
        #     self.transformed_clean_data = [trsf(c) for c in clean_data]  # 干净信号也应用相同变换
        # if ret_data:
        #     return data, targets, snrs, DummyDataset(data, targets, snrs, trsf, self.use_path)
        # else:
        #     return DummyDataset(data, targets, snrs, trsf, self.use_path)

        # 【核心修改 3】：获取分级干净信号 (目标 = min(输入SNR + 4, 18))
        clean_data = []
        for i in range(len(targets)):
            c = targets[i]
            s = snrs[i]
            # 计算分级目标 SNR
            target_s = 18

            # 从 ref_dict 里面随机挑一个同类别、目标SNR的参考信号
            available_refs = self.ref_dict.get((c, target_s))
            if available_refs is None or len(available_refs) == 0:
                # 理论上不会为空，万一为空，fallback到该类别的 18dB
                available_refs = self.ref_dict.get((c, 18.0))

            clean_sig = random.choice(available_refs)
            clean_data.append(clean_sig)

        clean_data = np.stack(clean_data, axis=0)

        # 存入实例变量供 icarl.py 调用
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            self.transformed_clean_data = [trsf(c) if trsf else c for c in clean_data]
        else:
            self.transformed_clean_data = clean_data

        if ret_data:
            return data, targets, snrs, DummyDataset(data, targets, snrs, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, snrs, trsf, self.use_path)

    def get_finetune_dataset(self, known_classes, total_classes, source, mode, appendent, type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx + 1)
            num = len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot * (total_classes - known_classes) / known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot / (total_classes - known_classes))
        for idx in range(known_classes, total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data = np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)


    def _setup_data(self, dataset_name, shuffle, seed):
        with open('RML2016.10a_dict.pkl', 'rb') as file:
            Xd = pickle.load(file, encoding='latin')

        snrs_all, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        # 【核心修改 1】：过滤出 -4dB 及以上的 SNR 列表
        self.snrs_valid = [s for s in snrs_all if s >= -4]

        X = []
        lbl = []
        for mod in mods:
            # 仅遍历 >= -4dB 的有效信噪比
            for snr in self.snrs_valid:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))

        X = np.vstack(X)
        n_examples = X.shape[0]
        n_train = int(0.75 * n_examples)

        allnum = list(range(0, n_examples))
        if shuffle:
            random.seed(seed)
            np.random.seed(seed)
            random.shuffle(allnum)

        train_idx = allnum[0:n_train]
        test_idx = allnum[n_train:]

        self._train_data = X[train_idx]
        self._train_targets = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
        self._train_snr = list(map(lambda x: lbl[x][1], train_idx))

        self._test_data = X[test_idx]
        self._test_targets = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
        self._test_snr = list(map(lambda x: lbl[x][1], test_idx))

        self.use_path = False
        self._train_trsf = []
        self._test_trsf = []
        self._common_trsf = []

        # 获取类别顺序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = list(set(self._train_targets))
        self._class_order = order
        logging.info(self._class_order)

        # 映射增量标签
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

        # 【核心修改 2】：构建一个庞大的参考字典 dict: (class_idx, snr) -> list of samples
        # 用于后续 +4dB 分级信号的极速检索
        self.ref_dict = {}
        for i, (c, s) in enumerate(zip(self._train_targets, self._train_snr)):
            if (c, s) not in self.ref_dict:
                self.ref_dict[(c, s)] = []
            self.ref_dict[(c, s)].append(self._train_data[i])

        # 构建 Random_Matrix (仅针对过滤后的 >= -4dB 数据)
        N_random_sample = 50
        self.Random_Matrix = np.zeros([len(mods) * len(self.snrs_valid), 2, 128 * N_random_sample])
        count = 0
        for i in range(len(mods)):
            for snr in self.snrs_valid:
                source_data = Xd[(mods[i], snr)]
                choice = np.random.choice(range(source_data.shape[0]), size=N_random_sample, replace=False)
                random_sample = source_data[choice]
                random_sample = random_sample.swapaxes(0, 1)
                random_sample = np.reshape(random_sample, [2, 128 * N_random_sample])
                self.Random_Matrix[count] = random_sample
                count += 1

    # def _setup_data(self, dataset_name, shuffle, seed):
    #     # ==========================================
    #     # 1. 加载训练集 (RML2016.10a) 并过滤 SNR >= -4dB
    #     # ==========================================
    #     print("Loading Training Data (RML2016_10aRician.pkl)...")
    #     with open('RML2016_10aRician.pkl', 'rb') as file:
    #         Xd_train = pickle.load(file, encoding='latin')
    #
    #     snrs_all, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd_train.keys())))), [1, 0])
    #
    #     # 【核心修改 1】：过滤出 -4dB 及以上的 SNR 列表
    #     self.snrs_valid = [s for s in snrs_all if s >= -4]
    #     print(f"Valid SNRs (>= -4dB): {self.snrs_valid}")
    #
    #     X_train_list = []
    #     lbl_train = []
    #     for mod in mods:
    #         # 【修改】：仅遍历 >= -4dB 的有效信噪比
    #         for snr in self.snrs_valid:
    #             X_train_list.append(Xd_train[(mod, snr)].astype(np.float32))
    #             for i in range(Xd_train[(mod, snr)].shape[0]):
    #                 lbl_train.append((mod, snr))
    #
    #     X_train_full = np.vstack(X_train_list)
    #     n_examples_train = X_train_full.shape[0]
    #
    #     # 取前75%作为训练数据
    #     n_train = int(0.75 * n_examples_train)
    #     allnum_train = list(range(0, n_examples_train))
    #
    #     if shuffle:
    #         random.seed(seed)
    #         np.random.seed(seed)
    #         random.shuffle(allnum_train)
    #
    #     train_idx = allnum_train[0:n_train]
    #
    #     self._train_data = X_train_full[train_idx]
    #     self._train_targets = list(map(lambda x: mods.index(lbl_train[x][0]), train_idx))
    #     self._train_snr = list(map(lambda x: lbl_train[x][1], train_idx))
    #
    #     # ==========================================
    #     # 2. 加载跨域测试集 (MATLAB AWGN)，同样只取 >= -4dB
    #     # ==========================================
    #     test_dataset_path = 'RML2016_10aRayleigh.pkl'
    #     print(f"Loading Cross-Domain Testing Data ({test_dataset_path})...")
    #     with open(test_dataset_path, 'rb') as file:
    #         Xd_test = pickle.load(file)
    #
    #     X_test_list = []
    #     lbl_test = []
    #
    #     for mod in mods:
    #         # 【修改】：跨域测试集也只测 >= -4dB 的数据
    #         for snr in self.snrs_valid:
    #             target_key = None
    #             for k in Xd_test.keys():
    #                 k_mod = k[0].decode('utf-8') if isinstance(k[0], bytes) else k[0]
    #                 m_str = mod.decode('utf-8') if isinstance(mod, bytes) else mod
    #
    #                 if k_mod == m_str and int(k[1]) == int(snr):
    #                     target_key = k
    #                     break
    #
    #             if target_key is not None:
    #                 X_test_list.append(Xd_test[target_key].astype(np.float32))
    #                 for i in range(Xd_test[target_key].shape[0]):
    #                     lbl_test.append((mod, snr))
    #             else:
    #                 print(f"Warning: Cross-domain test data not found for {m_str} at SNR {snr}")
    #
    #     X_test_full = np.vstack(X_test_list)
    #     n_examples_test = X_test_full.shape[0]
    #     allnum_test = list(range(0, n_examples_test))
    #
    #     self._test_data = X_test_full
    #     self._test_targets = list(map(lambda x: mods.index(lbl_test[x][0]), allnum_test))
    #     self._test_snr = list(map(lambda x: lbl_test[x][1], allnum_test))
    #
    #     self.use_path = False
    #     self._train_trsf = []
    #     self._test_trsf = []
    #     self._common_trsf = []
    #
    #     # ==========================================
    #     # 3. 确定类别顺序并映射增量标签
    #     # ==========================================
    #     order = [i for i in range(len(np.unique(self._train_targets)))]
    #     if shuffle:
    #         np.random.seed(seed)
    #         order = np.random.permutation(len(order)).tolist()
    #     else:
    #         order = list(set(self._train_targets))
    #     self._class_order = order
    #     logging.info(f"Class Order: {self._class_order}")
    #
    #     self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
    #     self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
    #
    #     # ==========================================
    #     # 4. 构建用于检索的高速字典与高信噪比Reference
    #     # ==========================================
    #     # 【核心修改 2】：引入你要求的庞大参考字典，用于快速检索
    #     self.ref_dict = {}
    #     for i, (c, s) in enumerate(zip(self._train_targets, self._train_snr)):
    #         if (c, s) not in self.ref_dict:
    #             self.ref_dict[(c, s)] = []
    #         self.ref_dict[(c, s)].append(self._train_data[i])
    #
    #     self.high_snr_ref = {}
    #     for class_id in np.unique(self._train_targets):
    #         class_indices = np.where(self._train_targets == class_id)[0]
    #         max_snr = max(self._train_snr[i] for i in class_indices)
    #         indices = [i for i in class_indices if self._train_snr[i] == max_snr]
    #         ref_idx = np.random.choice(indices)
    #         self.high_snr_ref[class_id] = self._train_data[ref_idx]
    #
    #     self.ref_signals = torch.tensor(
    #         np.array([self.high_snr_ref[i] for i in sorted(self.high_snr_ref.keys())]),
    #         dtype=torch.float32
    #     )
    #
    #     # ==========================================
    #     # 5. 构建对齐的 Random_Matrix (仅针对过滤后的数据)
    #     # ==========================================
    #     print("Building Aligned Random Matrix (>= -4dB)...")
    #     N_random_sample = 50
    #     # 【修改】：矩阵大小现在取决于 self.snrs_valid 的长度
    #     self.Random_Matrix = np.zeros([len(self._class_order) * len(self.snrs_valid), 2, 128 * N_random_sample], dtype=np.float32)
    #     count = 0
    #
    #     for new_class_idx in range(len(self._class_order)):
    #         original_mod_index = self._class_order[new_class_idx]
    #         mod_name = mods[original_mod_index]
    #
    #         # 【修改】：只遍历过滤后的信噪比
    #         for snr in self.snrs_valid:
    #             source_data = Xd_train[(mod_name, snr)]
    #             num_samples = source_data.shape[0]
    #
    #             choice = np.random.choice(range(num_samples), size=N_random_sample, replace=False)
    #             random_sample = source_data[choice]
    #             random_sample = random_sample.swapaxes(0, 1)
    #             random_sample = np.reshape(random_sample, [2, 128 * N_random_sample])
    #
    #             self.Random_Matrix[count] = random_sample
    #             count += 1
    #
    #     print("Setup completed for Zero-Shot Cross-Domain Test with >= -4dB Data.\n")
    def _select(self, x, y, snr, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        snr_return = np.array(snr)[idxes]
        return x_return, y[idxes], snr_return

    def _select_rmm(self, x, y, snr, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        snr_return = np.array(snr)[new_idxes]

        return x[new_idxes], y[new_idxes], snr_return

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


# class DummyDataset(Dataset):
#     def __init__(self, images, labels, trsf, use_path=False):
#         assert len(images) == len(labels), "Data size error!"
#         self.images = images
#         self.labels = labels
#         self.trsf = trsf
#         self.use_path = use_path
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         if self.use_path:
#             image = self.trsf(pil_loader(self.images[idx]))
#         else:
#             image = self.trsf(Image.fromarray(self.images[idx]))
#         label = self.labels[idx]
#
#         return idx, image, label
class DummyDataset(Dataset):
    def __init__(self, images, labels, snrs, trsf=None, use_path=False):
        """
        Args:
            images (numpy.ndarray or list): 输入数据，形状为 (num_samples, height, width, channels)
            labels (numpy.ndarray or list): 标签，形状为 (num_samples,)
            trsf (callable, optional): 用于数据转换的函数，默认为 None
        """
        assert len(images) == len(labels), "Data size error!"
        assert len(snrs) == len(images), "Data size error!"
        self.images = images
        self.labels = labels
        self.snrs = snrs
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取样本数据
        image = self.images[idx]
        label = self.labels[idx]
        snrs = self.snrs[idx]

        # 如果提供了转换操作 (例如 ToTensor, 归一化等)
        if self.trsf:
            image = self.trsf(image)
        else:
            # 如果没有提供转换操作，则直接将数据转为 Tensor
            image = torch.tensor(image, dtype=torch.float32)

        # 返回索引、图像和标签
        return idx, image, label, snrs


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _map_new_class_index_2(y, order):
    # 构建复数标签到新索引的字典（提高查找效率）
    class_to_idx = {cls: idx for idx, cls in enumerate(order)}

    # 检查是否存在缺失标签
    missing_labels = set(y) - class_to_idx.keys()
    if missing_labels:
        raise ValueError(f"标签中存在未在order中定义的复数: {missing_labels}")

    # 通过字典直接映射（避免低效的列表.index()）
    return np.array([class_to_idx[cls] for cls in y])


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
