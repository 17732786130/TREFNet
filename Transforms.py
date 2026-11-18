import numpy
import numpy as np
import torch
import random
import cv2


class Scale(object):
    """
    Resize the given image to a fixed scale
    """

    def __init__(self, wi, he):
        '''
        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    # modified from torchvision to add support for max size

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class Resize(object):
    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if not self.strict:
            size = random.choice(self.min_size)
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)

    def __call__(self, image, label):
        size = self.get_size(image.shape[:2])
        # print("origin", image.shape)
        image = cv2.resize(image, size)
        # print("resized", image.shape)
        # print('*'*20)
        # I confirm that the output size is right, not reversed
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return (image, label)


class RandomCropResize(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """

    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.cw)
            y1 = random.randint(0, self.ch)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            return img_crop, label_crop
        else:
            return [img, label]


class RandomFlip(object):
    """
    Randomly flip the given Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
                image = cv2.flip(image, 0)  # horizontal flip
                label = cv2.flip(label, 0)  # horizontal flip
        if random.random() < 0.5:
                image = cv2.flip(image, 1)  # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
        return [image, label]


class RandomExchange(object):
    """
    Randomly flip the given Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
        return [image, label]

# ======================== 新增：5个双时相同步增强类 ========================
class RandomColorJitter(object):
    """双时相同步颜色抖动 - 兼容(img, label)接口"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, label):
        # img: [H, W, 6]  前3=前时相，后3=后时相；label不变
        if random.random() < 0.5:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            img = np.clip(img * brightness_factor, 0, 1)
        if random.random() < 0.5:
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = np.mean(img, axis=(0,1), keepdims=True)
            img = np.clip(mean + contrast_factor * (img - mean), 0, 1)
        if random.random() < 0.5:
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            gray = np.mean(img, axis=2, keepdims=True)
            img = np.clip(gray + saturation_factor * (img - gray), 0, 1)
        if random.random() < 0.5 and self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            img = np.clip(img + hue_factor, 0, 1)
        return img, label

class RandomGaussianNoise(object):
    """双时相同步加高斯噪声 - 兼容(img, label)接口"""
    def __init__(self, std=0.01):
        self.std = std
    def __call__(self, img, label):
        if random.random() < 0.5:
            noise = np.random.normal(0, self.std, img.shape)
            img = np.clip(img + noise, 0, 1)
        return img, label

class RandomScale(object):
    """双时相同步随机缩放 - 兼容(img, label)接口"""
    def __init__(self, scale_range=(0.8, 1.2)):
        self.min_scale = scale_range[0]
        self.max_scale = scale_range[1]
    def __call__(self, img, label):
        H, W = img.shape[:2]
        scale = random.uniform(self.min_scale, self.max_scale)
        new_H, new_W = int(H * scale), int(W * scale)
        # 同步缩放img和label
        img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        return img, label

class RandomRotate(object):
    """双时相同步随机旋转 - 兼容(img, label)接口"""
    def __init__(self, angle_range=(-10, 10)):
        self.min_angle = angle_range[0]
        self.max_angle = angle_range[1]
    def __call__(self, img, label):
        if random.random() < 0.5:
            H, W = img.shape[:2]
            angle = random.uniform(self.min_angle, self.max_angle)
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
            # 同步旋转img和label
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            label = cv2.warpAffine(label, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        return img, label

class RandomBlur(object):
    """双时相同步随机模糊 - 兼容(img, label)接口"""
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
    def __call__(self, img, label):
        if random.random() < 0.5:
            kernel = np.ones((self.kernel_size, self.kernel_size), np.float32) / (self.kernel_size**2)
            img = cv2.filter2D(img, -1, kernel)
        return img, label

# ======================== 原有类继续保留 ========================
class Normalize(object):
    """
    Given mean: (B, G, R) and std: (B, G, R),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std
        self.depth_mean = [0.5]
        self.depth_std = [0.5]

    def __call__(self, image, label):
        image = image.astype(np.float32)
        image = image / 255
        label = np.ceil(label / 255)
        for i in range(6):
            image[:, :, i] -= self.mean[i]
        for i in range(6):
            image[:, :, i] /= self.std[i]

        return [image, label]


class GaussianNoise(object):
    def __init__(self, std=0.05):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.std = std

    def __call__(self, image, label):
        noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label]


class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''

    def __init__(self, scale=1):
        '''
        :param scale: set this parameter according to the output scale
        '''
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)), \
                               interpolation=cv2.INTER_NEAREST)
        image = image[:, :, ::-1].copy()  # .copy() is to solve "torch does not support negative index"
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        # TODO: here, we add unsqueeze to satisfy the condition that
        # adjust_size in DataSet.py should input 4D tensor
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int64)).unsqueeze(dim=0)

        return [image_tensor, label_tensor]


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args