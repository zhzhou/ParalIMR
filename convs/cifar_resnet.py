import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------
# 1D下采样模块（适配时序信号）
# ------------------------------
class Downsample1D(nn.Module):
    """1D下采样模块：用于残差块的维度匹配"""

    def __init__(self, nIn, nOut, stride):
        super(Downsample1D, self).__init__()
        self.conv = nn.Conv1d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# ------------------------------
# 1D残差块（适配IQ时序信号）
# ------------------------------
class ResNetBasicblock1D(nn.Module):
    """1D版本的基础残差块，用于时序特征提取"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock1D, self).__init__()
        # 第一个1D卷积：调整通道数和时序长度
        self.conv_a = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个1D卷积：保持通道数和时序长度
        self.conv_b = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm1d(planes)

        self.downsample = downsample  # 下采样模块（处理维度不匹配）

    def forward(self, x):
        residual = x  # 跳跃连接的残差分支

        # 主分支：卷积→批归一化→激活
        out = self.conv_a(x)
        out = self.bn_a(out)
        out = self.relu(out)

        out = self.conv_b(out)
        out = self.bn_b(out)

        # 若残差与主分支维度不匹配，用downsample调整
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接：主分支 + 残差分支
        out += residual
        out = self.relu(out)
        return out


# ------------------------------
# 主模型：1D降噪残差网络（适配IQ数据）
# ------------------------------
class DenoisingCifarResNet(nn.Module):
    """用于IQ时序数据的1D残差网络，带降噪预处理"""

    def __init__(self, block, depth, channels=2, seq_length=128, num_classes=11):
        super(DenoisingCifarResNet, self).__init__()

        # 1. 移除独立的denoiser，后续将ResNet主干作为共享编码器

        # 2. 残差网络参数校验
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        # 3. 初始卷积层（共享编码器的第一部分）
        self.conv_1_3x3 = nn.Conv1d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)

        # 4. 残差阶段（共享编码器的主干）
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, stride=1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, stride=2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, stride=2)

        # 5. 分类头 (一个分支)
        self.avgpool = nn.AvgPool1d(32)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 64 * block.expansion
        self.dropout = nn.Dropout(p=0.2)  # 新增Dropout层
        self.fc = nn.Linear(self.out_dim, num_classes)

        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)  # 32->64

        self.up2 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)  # 64->128
        self.out_conv = nn.Sequential(
            nn.Conv1d(16, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, channels, 3, 1, 1),
            nn.Tanh()
        )

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建1D残差块序列"""
        downsample = None
        # 若步长≠1（下采样）或通道数不匹配，创建下采样模块
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample1D(
                nIn=self.inplanes,
                nOut=planes * block.expansion,
                stride=stride
            )

        layers = []
        # 第一个残差块（可能带下采样）
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # 更新当前通道数

        # 后续残差块（无下采样）
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, return_decoded=True):
        # 共享编码器前向传播
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        x = self.relu(x)

        x_1 = self.stage_1(x)
        x_2 = self.stage_2(x_1)
        x_3 = self.stage_3(x_2)  # 这是共享编码器的最终输出特征 [batch, 64, 32]

        # attn_weights = self.attention(x_3)
        # x_3 = x_3 * attn_weights
        # --- 分类分支 ---
        pooled = self.avgpool(x_3)
        features = pooled.view(pooled.size(0), -1)
        features = self.dropout(features)
        # logits = self.fc(features)
        #
        # --- 重构分支 ---
        self.skip_weight = nn.Parameter(torch.tensor(0.3))
        # 将共享编码器的输出特征送入解码器

        d3 = self.relu(self.up3(x_3))
        d3 = d3 + 0.5 * x_2
        d2 = self.relu(self.up2(d3))
        decoded = self.out_conv(d2)

        result = {
            'fmaps': [x_1, x_2, x_3],
            'features': features
            #'logits': logits
        }

        if return_decoded:
            result['denoised'] = decoded # 注意这里key从'denoised'改为'decoded'更准确

        return result


    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b

# # 串联结构
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
#
# # ------------------------------
# # 1D下采样模块（适配时序信号）
# # ------------------------------
# class Downsample1D(nn.Module):
#     """1D下采样模块：用于残差块的维度匹配"""
#
#     def __init__(self, nIn, nOut, stride):
#         super(Downsample1D, self).__init__()
#         self.conv = nn.Conv1d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.bn = nn.BatchNorm1d(nOut)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x
#
#
# # ------------------------------
# # 1D残差块（适配IQ时序信号）
# # ------------------------------
# class ResNetBasicblock1D(nn.Module):
#     """1D版本的基础残差块，用于时序特征提取"""
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(ResNetBasicblock1D, self).__init__()
#         # 第一个1D卷积：调整通道数和时序长度
#         self.conv_a = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn_a = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 第二个1D卷积：保持通道数和时序长度
#         self.conv_b = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn_b = nn.BatchNorm1d(planes)
#
#         self.downsample = downsample  # 下采样模块（处理维度不匹配）
#
#     def forward(self, x):
#         residual = x  # 跳跃连接的残差分支
#
#         # 主分支：卷积→批归一化→激活
#         out = self.conv_a(x)
#         out = self.bn_a(out)
#         out = self.relu(out)
#
#         out = self.conv_b(out)
#         out = self.bn_b(out)
#
#         # 若残差与主分支维度不匹配，用downsample调整
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         # 残差连接：主分支 + 残差分支
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# # ------------------------------
# # 1D降噪自编码器（适配IQ信号）
# # ------------------------------
# class DenoisingAutoencoder1D(nn.Module):
#     """针对IQ时序数据的1D降噪自编码器"""
#
#     def __init__(self, in_channels=2, seq_length=128):
#         super(DenoisingAutoencoder1D, self).__init__()
#         self.in_channels = in_channels  # IQ数据通常为2通道
#         self.seq_length = seq_length  # 时序长度（如128）
#
#         # 编码器：压缩特征
#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels, 8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(8),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),  # 降采样到64
#             nn.BatchNorm1d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 降采样到32
#             nn.BatchNorm1d(32),
#             nn.ReLU(inplace=True),
#         )
#
#         # 解码器：重构信号
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 上采样到64
#             nn.BatchNorm1d(16),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),  # 上采样到128
#             nn.BatchNorm1d(8),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(8, in_channels, kernel_size=3, stride=1, padding=1),  # 恢复2通道
#             nn.Tanh()  # IQ信号通常在[-1,1]范围内
#         )
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         # x: (batch, 2, 128) → 噪声IQ信号
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)  # 输出降噪后的IQ信号
#         return decoded
#
#
# # ------------------------------
# # 主模型：1D降噪残差网络（适配IQ数据）
# # ------------------------------
# class DenoisingCifarResNet(nn.Module):
#     """用于IQ时序数据的1D残差网络，带降噪预处理"""
#
#     def __init__(self, block, depth, channels=2, seq_length=128, num_classes=11):
#         super(DenoisingCifarResNet, self).__init__()
#         # 1. 降噪模块：处理原始IQ信号
#         self.denoiser = DenoisingAutoencoder1D(in_channels=channels, seq_length=seq_length)
#
#         # 2. 残差网络参数校验（确保深度合法）
#         assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
#         layer_blocks = (depth - 2) // 6  # 每个stage的残差块数量
#
#         # 3. 初始卷积层（对降噪后的信号做初步特征提取）
#         self.conv_1_3x3 = nn.Conv1d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn_1 = nn.BatchNorm1d(16)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 4. 残差阶段（时序特征提取）
#         self.inplanes = 16  # 初始通道数
#         self.stage_1 = self._make_layer(block, 16, layer_blocks, stride=1)  # 保持长度128
#         self.stage_2 = self._make_layer(block, 32, layer_blocks, stride=2)  # 降采样到64
#         self.stage_3 = self._make_layer(block, 64, layer_blocks, stride=2)  # 降采样到32
#
#         # 5. 分类头
#         self.avgpool = nn.AvgPool1d(32)  # 对32长度的时序信号做全局池化
#         self.out_dim = 64 * block.expansion
#         self.fc = nn.Linear(64 * block.expansion, num_classes)
#
#         # 6. 权重初始化
#         self._initialize_weights()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         """构建1D残差块序列"""
#         downsample = None
#         # 若步长≠1（下采样）或通道数不匹配，创建下采样模块
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = Downsample1D(
#                 nIn=self.inplanes,
#                 nOut=planes * block.expansion,
#                 stride=stride
#             )
#
#         layers = []
#         # 第一个残差块（可能带下采样）
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion  # 更新当前通道数
#
#         # 后续残差块（无下采样）
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self):
#         """1D卷积权重初始化"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 m.bias.data.zero_()
#
#     def forward(self, x, return_denoised=True):
#         # x: (batch, 2, 128) → 输入含噪声的IQ信号
#         # 步骤1：降噪预处理
#         denoised_x = self.denoiser(x)
#         #
#         # # 步骤2：特征提取（基于降噪后的信号）
#         x = denoised_x.float()
#         x = self.conv_1_3x3(x)  # (batch, 16, 128)
#         x = self.bn_1(x)
#         x = self.relu(x)
#
#         x_1 = self.stage_1(x)  # (batch, 16, 128)
#         x_2 = self.stage_2(x_1)  # (batch, 32, 64)
#         x_3 = self.stage_3(x_2)  # (batch, 64, 32)
#
#         # 步骤3：分类
#         pooled = self.avgpool(x_3)  # (batch, 64, 1)
#         features = pooled.view(pooled.size(0), -1)  # (batch, 64)
#         # logits = self.fc(features)  # (batch, num_classes)
#
#         # 构建返回结果
#         result = {
#             'fmaps': [x_1, x_2, x_3],  # 各阶段特征图
#             'features': features  # 池化后的特征
#             # 'logits': logits  # 分类输出
#         }
#
#         # # 可选：返回降噪后的信号（用于可视化或损失计算）
#         if return_denoised:
#             result['denoised'] = denoised_x
#
#         return result
#
#     @property
#     def last_conv(self):
#         """返回最后一个卷积层（用于后续可视化或特征提取）"""
#         return self.stage_3[-1].conv_b


# ------------------------------
# 模型创建函数（与原文件接口保持一致）
# ------------------------------
def resnet20():
    """简化模型：替代原ResNet-20"""
    return DenoisingCifarResNet(depth=20, channels=2, num_classes=11)


def resnet32():
    """简化模型：替代原ResNet-32"""
    return DenoisingCifarResNet(ResNetBasicblock1D, 32)


def resnet44():
    return DenoisingCifarResNet(depth=44, channels=2, num_classes=11)


def resnet56():
    return DenoisingCifarResNet(depth=56, channels=2, num_classes=11)


def resnet110():
    return DenoisingCifarResNet(depth=110, channels=2, num_classes=11)


# 轻量化模型
def resnet14():
    return DenoisingCifarResNet(depth=14, channels=2, num_classes=11)


def resnet26():
    return DenoisingCifarResNet(depth=26, channels=2, num_classes=11)