import argparse
from backend.orchestrator import Orchestrator
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", required=True)
    ap.add_argument("-o", "--output_dir", required=True)

    print("Generating code...")
    args = vars(ap.parse_args())
    img_path = args["img_path"]
    output_dir = args["output_dir"]
    orchestrator = Orchestrator(img_path, output_dir)
    orchestrator.create_DSL(img_path)
    orchestrator.create_code()
    print("Generation complete!")
