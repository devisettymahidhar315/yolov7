import torch
import numpy as np
import logging
from loguru import logger
from models.yolov7.reference.yolov7_detect import detect, parse_opt
from models.yolov7.reference.yolov7_model import Model
from models.yolov7.reference.yolov7_utils import *

def test_demo():
    args = [
            '--weights', 'yolov7.pt',
            '--source', 'models/yolov7/tests/horses.jpg',
            '--conf-thres', '0.25',
            '--img-size', '640'
        ]
    opt = parse_opt(args)
    device = select_device(opt.device)
    inference_model = attempt_load("yolov7.pt", map_location=device)
    sys.modules['models.common'] = sys.modules['models.yolov7.reference.yolov7_utils']
    def load_weights(model, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
    reference_model = Model()
    weights_path = "models/yolov7/tests/yolov7.pt"
    load_weights(reference_model, weights_path)
    torch_output = detect(opt, inference_model)
    reference_output = detect(opt, reference_model)
    logger.info(f"Predicted Answer from Yolov7 model: {torch_output}")
    logger.info(f"Predicted Answer from Reference model: {reference_output}")