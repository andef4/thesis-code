from PIL import Image


def resize_image(paths, size):
    in_file, out_file = paths
    image = Image.open(in_file)
    rgb_image = Image.new('RGB', image.size)
    rgb_image.paste(image)
    rgb_image = rgb_image.resize((size, size), Image.NEAREST)
    rgb_image.save(out_file)

def resize_image224(paths):
    resize_image(paths, 224)

def resize_image299(paths):
    resize_image(path, 299)
