import random

from diagrams import Diagram
from diagrams.custom import Custom

import cv2
import os
from programming.DSLBuilder import DSLBuilder
from programming.util import generate_var_name


from programming.code_generator import generate_code


def sketchify_image(img_path, outpath=None, k_size=5):
    if outpath is None:
        outpath = img_path

    print("Image path: {}".format(img_path))
    img = cv2.imread(img_path)

    # Convert to Grey Image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img = cv2.bitwise_not(grey_img)
    # invert_img=255-grey_img

    # Blur image
    blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)

    # Invert Blurred Image
    invblur_img = cv2.bitwise_not(blur_img)
    # invblur_img=255-blur_img

    # Sketch Image
    sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)

    # Save Sketch
    cv2.imwrite(outpath, sketch_img)


def parse_var(var_name):
    var_name = var_name.split(".")[0]
    var_name = var_name.replace("-", "_").replace(" ", "_")
    return var_name


class DSLGenerator:
    def __init__(self, diagram_type="pipeline"):
        self.diagram_type = diagram_type
        self.dslBuilder = DSLBuilder()

    def generate_dsl(self, var_names, classes, filename):
        pipeline_id = "main_pipeline"  # generate_var_name(False, False, 0.0, True)

        if self.diagram_type == "pipeline":
            self.dslBuilder.build_pipeline(pipeline_id, classes, var_names)

        self.dslBuilder.save_yaml(filename)


class DiagramGenerator:
    def __init__(self, diagram_type, class_names, class_symbols_path):
        self.diagram_type = diagram_type
        self.class_names = class_names
        self.class_symbols_path = class_symbols_path

    def generate_random_diagram(self, nodes_count, filename, sketchify=False):
        if self.diagram_type == "pipeline":
            classes = random.choices(self.class_names, k=nodes_count)

            with Diagram("", show=False, filename=filename):
                # Choose random class
                node_class_paths = [
                    self.class_symbols_path + "/" + class_ + "/" for class_ in classes
                ]
                # From that class, select a random image
                symbols = [random.choice(os.listdir(node)) for node in node_class_paths]
                var_names = [generate_var_name() for _ in range(nodes_count)]
                nodes = [
                    Custom(
                        var_name,
                        node_class_path + symbol,
                    )
                    for var_name, node_class_path, symbol in zip(
                        var_names, node_class_paths, symbols
                    )
                ]

                for i in range(nodes_count - 1):
                    nodes[i] >> nodes[i + 1]

        if sketchify:
            sketchify_image(filename + ".png")

        return var_names, classes


class NotebookGenerator:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def generate_notebook(self, out_name, yaml_dsl):
        generate_code(self.output_folder, out_name, yaml_dsl)


# if __name__ == "__main__":
#     notebooks_path = "/prototype/"
#     nb_generator = NotebookGenerator(notebooks_path)
#     dsl_path = "/prototype/"
#     diag_name = "my_diag"
#     var_names = ["My Step 1", "My Step 2", "My Step 3"]
#     classes = ["NeuralNet", "Merge", "Projection"]
#     dsl_generator = DSLGenerator("pipeline")
#     dsl_generator.generate_dsl(var_names, classes, dsl_path + diag_name + ".yml")
#     nb_generator.generate_notebook(diag_name + ".ipynb", dsl_path + diag_name + ".yml")
