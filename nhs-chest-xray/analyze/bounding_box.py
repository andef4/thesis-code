import os
import csv
from PIL import ImageDraw

BBOX_CSV_FILE = os.path.join('..', 'data', 'BBox_List_2017.csv')
ORIGINAL = 1024
SCALE = 224
SCALE_FACTOR = SCALE / ORIGINAL

bboxes = None
def load_bboxes():
    global bboxes
    bboxes = {}
    with open(BBOX_CSV_FILE) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # skip header
        for row in reader:
            image, _, x, y, w, h, *_ = row
            bboxes[image] = {
                'x': float(x) * SCALE_FACTOR,
                'y': float(y) * SCALE_FACTOR,
                'w': float(w) * SCALE_FACTOR,
                'h': float(h) * SCALE_FACTOR,
            }

def draw_bbox(image_name, image):
    if not bboxes:
        load_bboxes()

    if image_name in bboxes:
        bbox = bboxes[image_name]
        draw = ImageDraw.Draw(image)
        draw.rectangle((bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), outline='red')
        return True
    return False
