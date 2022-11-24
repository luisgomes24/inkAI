import numpy as np
import pytesseract
import torch


from PIL import Image
from detectron2.engine import DefaultPredictor

from backend.config import label2id, id2label
import backend.util.coordinates as coord_util
import backend.util.image as img_util
from backend.util.transforms import Transforms


class ImageSegmenter:
    def __init__(self, symbol_cfg, text_cfg):
        self.symbol_cfg = symbol_cfg
        self.text_cfg = text_cfg
        self.symbol_predictor = DefaultPredictor(symbol_cfg)
        self.text_predictor = DefaultPredictor(text_cfg)

    def segment(self, image, output_file="output.png"):
        # Return segments ordered from left to right.

        outputs = self.symbol_predictor(image)

        segments = {
            "arrow": [],
            "symbol": [],
            "text": [],
        }

        classes = outputs["instances"].pred_classes
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores

        for c, b, s in zip(classes, boxes, scores):
            # TODO
            # Add Confidence threshold (Score=s)
            if id2label[c.item()] != "text":
                segments[id2label[c.item()]].append(b.to("cpu").numpy())

        # Symbols can have text:
        # We mask out the "symbols" and get text again
        masked_img = image
        for symbol_box in segments["symbol"]:
            masked_img = img_util.mask_image(masked_img, symbol_box)
        for symbol_box in segments["arrow"]:
            masked_img = img_util.mask_image(masked_img, symbol_box)

        # d = pytesseract.image_to_data(masked_img, output_type=pytesseract.Output.DICT)
        # n_boxes = len(d["level"])
        # for i in range(n_boxes):
        #     (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        #     x1, y1, x2, y2 = x, y, x + w, y + h
        #     segments["text"].append(np.array([x1, y1, x2, y2], dtype=np.float32))

        outputs = self.text_predictor(masked_img)
        classes = outputs["instances"].pred_classes
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        for c, b, s in zip(classes, boxes, scores):
            if id2label[c.item()] == "text":
                bbox = b.to("cpu").numpy()
                # if bbox[2] - bbox[0] < 1000:
                segments[id2label[c.item()]].append(bbox)

        segments["symbol_centers"] = [
            coord_util.get_box_center(s) for s in segments["symbol"]
        ]
        segments["text_centers"] = [
            coord_util.get_box_center(t) for t in segments["text"]
        ]
        segments["arrow_centers"] = [
            coord_util.get_box_center(a) for a in segments["arrow"]
        ]

        ordered_symbol_indexes = coord_util.sort_coordinates_x(
            segments["symbol_centers"]
        )
        segments["symbol"] = [segments["symbol"][i] for i in ordered_symbol_indexes]

        segments["symbol_label"] = []
        for index in ordered_symbol_indexes:
            try:
                label_index = coord_util.get_nearest_coord(
                    segments["symbol_centers"][index], segments["text_centers"]
                )
                segments["symbol_label"].append(segments["text"][label_index])
            except (Exception):
                pass
        image = np.array(img_util.draw_boxes(image, segments["symbol"], "red"))
        image = np.array(img_util.draw_boxes(image, segments["text"], "green"))
        image = np.array(img_util.draw_boxes(image, segments["arrow"], "blue"))
        img_util.save_image(output_file, image)

        return segments


class TextExtractor:
    def __init__(self):
        self.predictor = pytesseract.image_to_string

    def get_text(self, np_image):
        return (
            self.predictor(np_image)
            .replace(" ", "")
            .replace("\n", "")
            .replace("\f", "")
        )


class SymbolClassifier:
    def __init__(self):
        self.predictor = lambda image: self._predictor(image)
        self.class_names = [
            "BarPlot",
            "Loading",
            "Merge",
            "NeuralNet",
            "Projection",
            "ScatterPlot",
        ]

    def classify(self, np_image, level):
        model = torch.load(
            "models/classifier_model.pt", map_location=torch.device("cpu")
        )

        transform = Transforms()

        image = Image.fromarray(np_image)
        image = transform(image)
        outputs = model(image[None])

        _, predicted = torch.max(outputs, 1)

        return self.class_names[predicted.item()]
