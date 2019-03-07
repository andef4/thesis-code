from PIL import Image

def resize_image(paths):
    in_file, out_file = paths
    image = Image.open(in_file)
    rgb_image = Image.new('RGB', image.size)
    rgb_image.paste(image)
    rgb_image = rgb_image.resize((299, 299), Image.NEAREST)
    rgb_image.save(out_file)
