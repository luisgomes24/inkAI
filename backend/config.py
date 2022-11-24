from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
import os


def add_vit_config(cfg):
    """
    Add config for VIT.
    """
    _C = cfg

    _C.MODEL.VIT = CN()

    # CoaT model name.
    _C.MODEL.VIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]

    _C.MODEL.VIT.IMG_SIZE = [224, 224]

    _C.MODEL.VIT.POS_TYPE = "shared_rel"

    _C.MODEL.VIT.DROP_PATH = 0.0

    _C.MODEL.VIT.MODEL_KWARGS = "{}"

    _C.SOLVER.OPTIMIZER = "ADAMW"

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0

    _C.AUG = CN()

    _C.AUG.DETR = False


label2id = {
    "arrow": 0,
    "symbol": 1,
    "text": 2,
}

id2label = {
    0: "arrow",
    1: "symbol",
    2: "text",
}


def get_symbols_config():
    cfg = get_cfg()
    add_vit_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(
        "dit/object_detection/publaynet_configs/cascade/cascade_dit_base.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(id2label.values()))

    # Validation
    cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = "models"
    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "dit_fine_tuned.pt"
    )  # path to the model we just trained
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def get_text_config():
    cfg = get_cfg()
    add_vit_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(
        "dit/object_detection/publaynet_configs/cascade/cascade_dit_base.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = (
        "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth"
    )
    cfg.DATASETS.TRAIN = ("diagram_train",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = (
        2  # This is the real "batch size" commonly known to deep learning people
    )
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        list(id2label.values())
    )  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    # Validation
    cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TEST = ("diagram_val", )
    # cfg.TEST.EVAL_PERIOD = 1
    # cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.OUTPUT_DIR = "models"

    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "dit_fine_tuned.pt"
    )  # path to the model we just trained

    # cfg.MODEL.WEIGHTS = os.path.join(
    #     cfg.OUTPUT_DIR, "model_final.pth"
    # )  # path to the model we just trained

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.MODEL.DEVICE = "cpu"
    return cfg
