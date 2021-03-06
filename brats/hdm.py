from PIL import Image, ImageDraw
import numpy as np
from torchvision.transforms import ToTensor, Normalize
from scipy.spatial.distance import directed_hausdorff
import torch
import matplotlib.pyplot as plt

# +
RAW = 1  # the actual hausdorff distances as a 2D array
BETTER_ONLY = 2  # only return points which decrease the distance when convered
WORSE_ONLY = 3  # only return points which increase the distance when convered
ABSOLUTE = 4  # treat better and worse distances as the same
BASELINE_DIFF = 5


class HDMResult:
    def __init__(self, distances, baseline, image_width, image_height,
                 circle_size, offset):
        self.results = distances
        self.baseline = baseline
        self.width = image_width
        self.height = image_height
        self.circle_size = circle_size
        self.offset = offset

    def distances(self, result_type):
        if result_type == RAW:
            return self.results
        elif result_type == BETTER_ONLY:
            return self.baseline - np.clip(self.results, self.results.min(), self.baseline)
        elif result_type == WORSE_ONLY:
            return np.clip(self.results, self.baseline, self.results.max()) - self.baseline
        elif result_type == ABSOLUTE:
            zero_centered = self.results - self.baseline
            return np.abs(zero_centered)
        elif result_type == BASELINE_DIFF:
            return self.baseline - self.results
        else:
            raise ValueError('Invalid result_type, only hdm.RAW, hdm.ABSOLUTE, '
                             'hdm.BETTER_ONLY and hdm.WORSE_ONLY are supported.')

    def circle_map(self, result_type, color_map='Reds'):
        distances = self.distances(result_type)
        normalized = distances - distances.min()
        normalized = normalized / normalized.max()

        hdm_image = Image.new('RGB', (self.width, self.height), 0)
        draw = ImageDraw.Draw(hdm_image)

        color_map = plt.get_cmap(color_map)

        # returns indices which would sort the array in 1d shape
        sorted_indices = np.argsort(normalized, axis=None)

        # converts the 1d shaped sort indices into 2d indices
        sorted_indices = np.unravel_index(sorted_indices, normalized.shape)

        # convert tuples returned above into numpy array
        sorted_indices = np.dstack(sorted_indices)
        for x, y in sorted_indices[0]:
            color = color_map(normalized[x][y])
            color = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            draw.ellipse(
                [
                    (x * self.offset,
                     y * self.offset),
                    (x * self.offset + self.circle_size,
                     y * self.offset + self.circle_size),
                ],
                fill=color,
            )
        hdm_image = hdm_image.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)
        return hdm_image


class HausdorffDistanceMasks:
    def __init__(self, image_width, image_height):
        self.width = image_width
        self.height = image_height

    def generate_masks(self, circle_size, offset, normalize=False):
        self.circle_size = circle_size
        self.offset = offset
        self.x_count = int(self.width / offset)
        self.y_count = int(self.height / offset)

        self.masks = []
        for y_offset in range(self.y_count):
            row = []
            for x_offset in range(self.x_count):
                x = (x_offset * offset)
                y = (y_offset * offset)
                image = Image.new('L', (self.width, self.height), 255)
                draw = ImageDraw.Draw(image)
                draw.ellipse([(x, y), (x + circle_size, y + circle_size)], fill=0)
                tensor = ToTensor()(image)
                if normalize:
                    tensor = Normalize([0.5], [0.5])(tensor)
                tensor = tensor.squeeze()
                row.append(tensor)
            self.masks.append(row)

    def distance(self, model, image, segment, device):
        batch = image.unsqueeze(0)
        batch = batch.to(device)
        output = model(batch)
        output = output.detach().cpu().numpy()[0]
        return self.calculate_distance(output, segment)

    def calculate_distance(self, output, segment):
        hd1 = directed_hausdorff(output, segment)[0]
        hd2 = directed_hausdorff(segment, output)[0]
        return np.max([hd1, hd2])

    def apply_mask(self, mask, image):
        return torch.min(image, mask)

    def explain(self, model, image, segment, device, channel=-1):
        assert len(image.shape) == 3
        assert image.shape[1] == self.width
        assert image.shape[2] == self.height

        distances = np.zeros((self.y_count, self.x_count))

        baseline = self.distance(model, image, segment, device)

        for y_offset in range(self.y_count):
            for x_offset in range(self.x_count):
                mask = self.masks[x_offset][y_offset]
                if channel == -1:
                    # all channels
                    masked_image = self.apply_mask(image, mask)
                else:
                    masked_image = image.clone()
                    masked_image[channel] = self.apply_mask(image[channel], mask)
                distances[x_offset][y_offset] = self.distance(
                    model, masked_image, segment, device
                )
        # we copy all the state variables into the result so
        # the result is independent from the HausdorffDistanceMasks instance,
        # which could be reused with different parameters/masks
        return HDMResult(distances, baseline, self.width, self.height,
                         self.circle_size, self.offset)
