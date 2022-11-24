import cv2
from PIL import Image, ImageDraw
import os


def load_image(image_path):
    im = cv2.imread(image_path)
    return im


def crop_image(np_image, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    cropped_image = np_image[y1:y2, x1:x2]

    return cropped_image


def save_image(filepath, np_image):
    im = Image.fromarray(np_image)
    im.save(filepath)


def draw_boxes(np_image, bboxes, color):
    image = Image.fromarray(np_image)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=3)

    return image


def mask_image(image, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    masked = image.copy()  # initialize mask
    masked[y1:y2, x1:x2] = 255

    return masked


def convert_to_RGBA(img_path):
    for image in os.listdir(img_path):
        im = Image.open(image)
        # If is png image
        if im.format is "PNG":
            # and is not RGBA
            if im.mode is not "RGBA":
                im.convert("RGBA").save(f"{image}.png")
