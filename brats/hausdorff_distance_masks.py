from PIL import Image, ImageDraw
import numpy as np
from torchvision.transforms import ToTensor, Normalize
from scipy.spatial.distance import directed_hausdorff
import torch


class HausdorffDistanceMasks:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def generate_masks(self, circle_size, offset, normalize=False):
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

    def evaluate(self, image, segment, model, device):
        distances = np.zeros((self.y_count, self.x_count))

        for y_offset in range(self.y_count):
            for x_offset in range(self.x_count):
                mask = self.masks[x_offset][y_offset]
                mask = mask.to(device)
                masked_image = torch.min(image, mask)
                output = model(masked_image)
                output = output.detach().cpu().numpy()[0]
                hd1 = directed_hausdorff(output, segment)
                hd2 = directed_hausdorff(segment, output)
                distances[x_offset][y_offset] = np.max([hd1, hd2])
        return distances
