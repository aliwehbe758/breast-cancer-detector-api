import numpy as np
import torch
import torch.nn as nn
from PIL import Image

class ResizeWithPad(nn.Module):
    def __init__(self, new_shape, float_output=False):
        super(ResizeWithPad, self).__init__()
        self.new_shape = new_shape
        self.float_output = float_output

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = self.resize_with_pad(image, self.new_shape, self.float_output)

        return Image.fromarray(image)

    def resize_with_pad(self, image, new_shape, float_output=False):
        original_shape = (image.shape[1], image.shape[0])
        ratio_width = float(new_shape[0]) / original_shape[0]
        ratio_height = float(new_shape[1]) / original_shape[1]
        ratio = min(ratio_width, ratio_height)
        new_size = tuple([round(x * ratio) for x in original_shape])

        if ratio_width > 1 and ratio_height > 1:
            result = image
            delta_w = new_shape[0] - original_shape[0]
            delta_h = new_shape[1] - original_shape[1]
        else:
            result = self.resize_channels(image, new_size, filter_name="bilinear")
            delta_w = new_shape[0] - new_size[0]
            delta_h = new_shape[1] - new_size[1]

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        result_padding = np.pad(result, ((top, bottom), (left, right), (0, 0)), mode='constant')
        if not float_output:
            result_padding = np.uint8(np.rint(result_padding.clip(0., 255.)))

        return result_padding

    def resize_single_channel(self, x_np, output_size, filter_name="bicubic"):
        s1, s2 = output_size
        img = Image.fromarray(x_np.astype(np.float32), mode='F')
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    def resize_channels(self, x, new_size, filter_name="bilinear"):
        x = [self.resize_single_channel(x[:, :, idx], new_size, filter_name=filter_name) for idx in range(x.shape[2])]
        x = np.concatenate(x, axis=2).astype(np.float32)
        return x


class CustomNormalize(nn.Module):
    def __init__(self):
        super(CustomNormalize, self).__init__()

    def forward(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected input to be a torch.Tensor")

        return self.normalize(image)

    def normalize(self, image):
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
        return normalized_image