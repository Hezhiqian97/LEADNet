import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import cvtColor, preprocess_input


class NET_Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train=True, dataset_path=""):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        name = self.annotation_lines[index].split()[0]
        img_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg")
        label_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        # 读取图像 (BGR 转 RGB)
        image = cv2.imread(img_path)[:, :, ::-1]

        # 读取灰度标签图
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 👑 执行专为 1 像素微弱缺陷定制的安全数据增强
        image, label = self._augment(image, label, self.input_shape, self.train)

        # 图像归一化并调整通道顺序 (H, W, C) -> (C, H, W)
        image = preprocess_input(image.astype(np.float32))
        image = np.transpose(image, (2, 0, 1))

        # 处理超出类别的标签，将其归为背景或忽略类
        label[label >= self.num_classes] = self.num_classes

        # 生成 One-hot 编码的标签，用于计算 Dice Loss 等宏观重合度指标
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[label.reshape(-1)]
        seg_labels = seg_labels.reshape((*self.input_shape, self.num_classes + 1))

        return image, label, seg_labels

    def _rand(self, a=0., b=1.):
        return np.random.rand() * (b - a) + a

    def _augment(self, image, label, target_shape, randomize=True, hue=0.1, sat=0.7, val=0.3):
        """
        👑 针对 1 像素工业划痕定制的安全增强 (Safe Augmentation)
        【核心修改】：彻底废除会导致 1 像素标签湮灭的随机缩放和长宽比扭曲。
        只保留绝对安全的几何变换：等比例居中、安全翻转、随机平移游走。
        """
        h, w = target_shape
        ih, iw = image.shape[:2]

        # 1. 计算等比缩放系数 (绝对保证不改变划痕的物理长宽比例和拓扑连续性)
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        # 使用安全的插值算法将原图放入目标画幅
        # 图像用三次插值保持平滑，标签必须用最近邻插值保持类别整数！
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # 如果是验证集 (randomize=False)，直接居中放置，不加任何随机性
        if not randomize:
            new_image = np.full((h, w, 3), 128, dtype=np.uint8)
            new_label = np.zeros((h, w), dtype=np.uint8)
            dx, dy = (w - nw) // 2, (h - nh) // 2

            new_image[dy:dy + nh, dx:dx + nw, :] = image
            new_label[dy:dy + nh, dx:dx + nw] = label
            return new_image, new_label

        # ==========================================
        # 训练集安全增强开始
        # ==========================================

        # 2. 随机翻转 (绝对安全，不破坏 1 像素的线宽)
        if self._rand() < 0.5:
            image = cv2.flip(image, 1)  # 水平翻转
            label = cv2.flip(label, 1)
        if self._rand() < 0.5:
            image = cv2.flip(image, 0)  # 垂直翻转
            label = cv2.flip(label, 0)

        # 3. 随机平移放置 (替代了致命的随机缩放，依然能增加模型对缺陷位置的鲁棒性)
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        new_label = np.zeros((h, w), dtype=np.uint8)

        # 在目标画布内随机游走寻找安放起点
        dx = np.random.randint(0, w - nw + 1) if w > nw else 0
        dy = np.random.randint(0, h - nh + 1) if h > nh else 0

        # 将等比缩放后的图像安全地放入画布
        new_image[dy:dy + nh, dx:dx + nw, :] = image
        new_label[dy:dy + nh, dx:dx + nw] = label

        # 4. 色彩抖动 (HSV 空间的亮度、饱和度、色相扰动，对抵抗工业现场光照变化极其有效)
        r = np.random.uniform(-1, 1, 3) * np.array([hue, sat, val]) + 1
        hsv = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return new_image, new_label


def dataset_collate(batch):
    """
    DataLoader 的拼包函数 (Collate Function)
    负责将单张图片和标签打包成一个 Batch 张量传给 GPU
    """
    images, labels, seg_labels = zip(*batch)

    images = torch.from_numpy(np.array(images, dtype=np.float32))
    labels = torch.from_numpy(np.array(labels, dtype=np.int64))
    seg_labels = torch.from_numpy(np.array(seg_labels, dtype=np.float32))

    return images, labels, seg_labels