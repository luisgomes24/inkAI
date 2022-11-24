import os
import sys

from programming.code_generator import generate_code
import backend.util.image as img_util
from programming.DSLBuilder import DSLBuilder
from backend.Segmenter import ImageSegmenter, SymbolClassifier, TextExtractor
from backend.config import get_symbols_config, get_text_config


from dit.object_detection.ditod import add_vit_config


class Orchestrator:
    def __init__(self, img_path, output_dir):
        self.img_path = img_path
        self.output_dir = output_dir
        self.dslBuilder = DSLBuilder()
        self.symbol_classifier = SymbolClassifier()
        self.text_extractor = TextExtractor()

    def create_DSL(self, img_path):
        symbols_cfg = get_symbols_config()
        text_cfg = get_text_config()
        segmenter = ImageSegmenter(symbols_cfg, text_cfg)

        # for img_path in glob.glob(self.input_dir + "/*.png"):
        img_filename = os.path.basename(img_path)
        print("Creating pipeline", img_filename)

        im = img_util.load_image(img_path)
        result_file = os.path.join(
            self.output_dir,
            os.path.splitext(img_filename)[0] + "_result.png",
        )
        segments = segmenter.segment(im, output_file=result_file)
        symbols_class, text_labels = self._get_classes_and_labels(im, segments)
        pipeline_id = os.path.splitext(img_filename)[0]

        self.dslBuilder.build_pipeline(pipeline_id, symbols_class, text_labels)

        self.dslBuilder.save_yaml(
            os.path.join(self.output_dir, os.path.splitext(img_filename)[0]) + ".yaml"
        )

    def create_code(self):
        out_name = os.path.splitext(os.path.basename(self.img_path))[0] + ".ipynb"
        input_dsl_file = (
            self.output_dir
            + os.path.splitext(os.path.basename(self.img_path))[0]
            + ".yaml"
        )

        generate_code(self.output_dir, out_name=out_name, input_dsl_file=input_dsl_file)

    def _get_classes_and_labels(self, im, segments):

        symbols_class = []
        text_labels = []
        for bbox in segments["symbol"]:
            cropped_img = img_util.crop_image(im, bbox)
            symbols_class.append(self.symbol_classifier.classify(cropped_img, 0))

        for bbox in segments["symbol_label"]:
            cropped_img = img_util.crop_image(im, bbox)
            text_labels.append(self.text_extractor.get_text(cropped_img))

        return symbols_class, text_labels
