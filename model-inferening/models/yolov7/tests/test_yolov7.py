import torch
import numpy as np
from loguru import logger
from models.yolov7.reference.yolov7_detect import detect, parse_opt
from models.yolov7.reference.yolov7_model import *
from models.yolov7.reference.yolov7_utils import *
from models.helper_funcs import comp_pcc


def test_model():
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
    if isinstance(reference_output, list):
        ref_output = [tensor for tensor in reference_output]
    else:
        ref_output = [reference_output]
    if isinstance(torch_output, list):
        torch_output = [tensor for tensor in torch_output]
    else:
        torch_output = [torch_output]
    for ref_tensor, torch_tensor in zip(ref_output, torch_output):
        pcc_value = comp_pcc(ref_tensor, torch_tensor)
        assert pcc_value[0], pcc_value[1]


def test_conv():
    sys.modules['models.common'] = sys.modules['models.yolov7.reference.yolov7_utils']
    torch_model = attempt_load("yolov7.pt", map_location="cpu")
    torch_layer = torch_model.model[0]
    conv_layer = Conv(c1=3, c2=32, k=3, s=1, p=1, act=True)
    initialize_weights(conv_layer)
    checkpoint = torch.load("models/yolov7/tests/yolov7.pt", map_location="cpu")
    state_dict = checkpoint["model"].state_dict()
    conv_layer.conv.weight.data.copy_(state_dict["model.0.conv.weight"].float())
    conv_layer.bn.weight.data.copy_(state_dict["model.0.bn.weight"].float())
    conv_layer.bn.bias.data.copy_(state_dict["model.0.bn.bias"].float())
    conv_layer.bn.running_mean.copy_(state_dict["model.0.bn.running_mean"].float())
    conv_layer.bn.running_var.copy_(state_dict["model.0.bn.running_var"].float())
    conv_layer.bn.num_batches_tracked.copy_(
        state_dict["model.0.bn.num_batches_tracked"].float()
    )
    conv_layer.eval()
    torch_layer.eval()
    torch.manual_seed(42)
    input = torch.rand((1, 3, 64, 64))
    with torch.inference_mode():
        reference_output = conv_layer(input)
        torch_output = torch_layer(input)
    pcc_value = comp_pcc(reference_output, torch_output)
    assert pcc_value[0], pcc_value[1]
    logger.info(f"Pcc value for Conv Layer: {pcc_value}")


def test_Mp():
    sys.modules['models.common'] = sys.modules['models.yolov7.reference.yolov7_utils']
    mp = MP()
    model = attempt_load("yolov7.pt", map_location="cpu")
    mo = model.model[12]
    torch.manual_seed(42)
    input = torch.rand((1, 3, 64, 64))
    with torch.inference_mode():
        reference_output = mp(input)
        torch_output = mo(input)
    pcc_value = comp_pcc(reference_output, torch_output)
    assert pcc_value[0], pcc_value[1]
    logger.info(f"Pcc value for Mp Layer: {pcc_value}")


def test_SPPCSPC():
    sys.modules['models.common'] = sys.modules['models.yolov7.reference.yolov7_utils']
    torch_model = attempt_load("yolov7.pt", map_location="cpu")
    torch_layer = torch_model.model[51]
    reference_layer = SPPCSPC(1024, 512)
    initialize_weights(reference_layer)
    checkpoint = torch.load("models/yolov7/tests/yolov7.pt", map_location="cpu")
    state_dict = checkpoint["model"].state_dict()
    reference_layer.cv1.conv.weight.data.copy_(
        state_dict["model.51.cv1.conv.weight"].float()
    )
    reference_layer.cv1.bn.weight.data.copy_(
        state_dict["model.51.cv1.bn.weight"].float()
    )
    reference_layer.cv1.bn.bias.data.copy_(state_dict["model.51.cv1.bn.bias"].float())
    reference_layer.cv1.bn.running_mean.copy_(
        state_dict["model.51.cv1.bn.running_mean"].float()
    )
    reference_layer.cv1.bn.running_var.copy_(
        state_dict["model.51.cv1.bn.running_var"].float()
    )
    reference_layer.cv1.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv1.bn.num_batches_tracked"].float()
    )
    reference_layer.cv2.conv.weight.data.copy_(
        state_dict["model.51.cv2.conv.weight"].float()
    )
    reference_layer.cv2.bn.weight.data.copy_(
        state_dict["model.51.cv2.bn.weight"].float()
    )
    reference_layer.cv2.bn.bias.data.copy_(state_dict["model.51.cv2.bn.bias"].float())
    reference_layer.cv2.bn.running_mean.copy_(
        state_dict["model.51.cv2.bn.running_mean"].float()
    )
    reference_layer.cv2.bn.running_var.copy_(
        state_dict["model.51.cv2.bn.running_var"].float()
    )
    reference_layer.cv2.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv2.bn.num_batches_tracked"].float()
    )
    reference_layer.cv3.conv.weight.data.copy_(
        state_dict["model.51.cv3.conv.weight"].float()
    )
    reference_layer.cv3.bn.weight.data.copy_(
        state_dict["model.51.cv3.bn.weight"].float()
    )
    reference_layer.cv3.bn.bias.data.copy_(state_dict["model.51.cv3.bn.bias"].float())
    reference_layer.cv3.bn.running_mean.copy_(
        state_dict["model.51.cv3.bn.running_mean"].float()
    )
    reference_layer.cv3.bn.running_var.copy_(
        state_dict["model.51.cv3.bn.running_var"].float()
    )
    reference_layer.cv3.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv3.bn.num_batches_tracked"].float()
    )
    reference_layer.cv4.conv.weight.data.copy_(
        state_dict["model.51.cv4.conv.weight"].float()
    )
    reference_layer.cv4.bn.weight.data.copy_(
        state_dict["model.51.cv4.bn.weight"].float()
    )
    reference_layer.cv4.bn.bias.data.copy_(state_dict["model.51.cv4.bn.bias"].float())
    reference_layer.cv4.bn.running_mean.copy_(
        state_dict["model.51.cv4.bn.running_mean"].float()
    )
    reference_layer.cv4.bn.running_var.copy_(
        state_dict["model.51.cv4.bn.running_var"].float()
    )
    reference_layer.cv4.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv4.bn.num_batches_tracked"].float()
    )
    reference_layer.cv5.conv.weight.data.copy_(
        state_dict["model.51.cv5.conv.weight"].float()
    )
    reference_layer.cv5.bn.weight.data.copy_(
        state_dict["model.51.cv5.bn.weight"].float()
    )
    reference_layer.cv5.bn.bias.data.copy_(state_dict["model.51.cv5.bn.bias"].float())
    reference_layer.cv5.bn.running_mean.copy_(
        state_dict["model.51.cv5.bn.running_mean"].float()
    )
    reference_layer.cv5.bn.running_var.copy_(
        state_dict["model.51.cv5.bn.running_var"].float()
    )
    reference_layer.cv5.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv5.bn.num_batches_tracked"].float()
    )
    reference_layer.cv6.conv.weight.data.copy_(
        state_dict["model.51.cv6.conv.weight"].float()
    )
    reference_layer.cv6.bn.weight.data.copy_(
        state_dict["model.51.cv6.bn.weight"].float()
    )
    reference_layer.cv6.bn.bias.data.copy_(state_dict["model.51.cv6.bn.bias"].float())
    reference_layer.cv6.bn.running_mean.copy_(
        state_dict["model.51.cv6.bn.running_mean"].float()
    )
    reference_layer.cv6.bn.running_var.copy_(
        state_dict["model.51.cv6.bn.running_var"].float()
    )
    reference_layer.cv6.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv6.bn.num_batches_tracked"].float()
    )
    reference_layer.cv7.conv.weight.data.copy_(
        state_dict["model.51.cv7.conv.weight"].float()
    )
    reference_layer.cv7.bn.weight.data.copy_(
        state_dict["model.51.cv7.bn.weight"].float()
    )
    reference_layer.cv7.bn.bias.data.copy_(state_dict["model.51.cv7.bn.bias"].float())
    reference_layer.cv7.bn.running_mean.copy_(
        state_dict["model.51.cv7.bn.running_mean"].float()
    )
    reference_layer.cv7.bn.running_var.copy_(
        state_dict["model.51.cv7.bn.running_var"].float()
    )
    reference_layer.cv7.bn.num_batches_tracked.copy_(
        state_dict["model.51.cv7.bn.num_batches_tracked"].float()
    )
    reference_layer.eval()
    torch_layer.eval()
    torch.manual_seed(42)
    input = torch.rand((1, 1024, 64, 64))
    with torch.inference_mode():
        reference_output = reference_layer(input)
        torch_output = torch_layer(input)
    pcc_value = comp_pcc(reference_output, torch_output)
    assert pcc_value[0], pcc_value[1]
    logger.info(f"Pcc value for SPPCSPC Layer: {pcc_value}")


def test_RepConv():
    sys.modules['models.common'] = sys.modules['models.yolov7.reference.yolov7_utils']
    torch_model = attempt_load("yolov7.pt", map_location="cpu")
    torch_layer = torch_model.model[102]
    reference_layer = RepConv(128, 256)
    initialize_weights(reference_layer)
    checkpoint = torch.load("models/yolov7/tests/yolov7.pt", map_location="cpu")
    state_dict = checkpoint["model"].state_dict()
    with torch.no_grad():
        reference_layer.rbr_dense[0].weight.copy_(
            state_dict["model.102.rbr_dense.0.weight"].float()
        )
        reference_layer.rbr_dense[1].weight.copy_(
            state_dict["model.102.rbr_dense.1.weight"].float()
        )
        reference_layer.rbr_dense[1].bias.copy_(
            state_dict["model.102.rbr_dense.1.bias"].float()
        )
        reference_layer.rbr_dense[1].running_mean.copy_(
            state_dict["model.102.rbr_dense.1.running_mean"].float()
        )
        reference_layer.rbr_dense[1].running_var.copy_(
            state_dict["model.102.rbr_dense.1.running_var"].float()
        )
        reference_layer.rbr_dense[1].num_batches_tracked.copy_(
            state_dict["model.102.rbr_dense.1.num_batches_tracked"]
        )
        reference_layer.rbr_1x1[0].weight.copy_(
            state_dict["model.102.rbr_1x1.0.weight"].float()
        )
        reference_layer.rbr_1x1[1].weight.copy_(
            state_dict["model.102.rbr_1x1.1.weight"].float()
        )
        reference_layer.rbr_1x1[1].bias.copy_(
            state_dict["model.102.rbr_1x1.1.bias"].float()
        )
        reference_layer.rbr_1x1[1].running_mean.copy_(
            state_dict["model.102.rbr_1x1.1.running_mean"].float()
        )
        reference_layer.rbr_1x1[1].running_var.copy_(
            state_dict["model.102.rbr_1x1.1.running_var"].float()
        )
        reference_layer.rbr_1x1[1].num_batches_tracked.copy_(
            state_dict["model.102.rbr_1x1.1.num_batches_tracked"]
        )

    torch.manual_seed(42)
    input = torch.rand((1, 128, 64, 64))
    reference_layer.eval()
    torch_layer.eval()
    with torch.inference_mode():
        reference_output = reference_layer(input)
        torch_output = torch_layer(input)
    pcc_value = comp_pcc(reference_output, torch_output)
    assert pcc_value[0], pcc_value[1]
    logger.info(f"Pcc value for RepConv Layer: {pcc_value}")

def test_Detect():
    sys.modules['models.common'] = sys.modules['models.yolov7.reference.yolov7_utils']
    torch_layer = attempt_load("yolov7.pt", map_location="cpu")
    torch_model = torch_layer.model[105]
    anchors = [
    [12, 16, 19, 36, 40, 28],  
    [36, 75, 76, 55, 72, 146],  
    [142, 110, 192, 243, 459, 401]  
    ]

    ch = [256, 512, 1024]  
    reference_model = Detect(nc=80, anchors=anchors, ch=ch)

    checkpoint = torch.load("models/yolov7/tests/yolov7.pt", map_location='cpu')
    state_dict = checkpoint['model'].state_dict()

    reference_model.m[0].weight.data.copy_(state_dict['model.105.m.0.weight'].float())
    reference_model.m[0].bias.data.copy_(state_dict['model.105.m.0.bias'].float())

    reference_model.m[1].weight.data.copy_(state_dict['model.105.m.1.weight'].float())
    reference_model.m[1].bias.data.copy_(state_dict['model.105.m.1.bias'].float())

    reference_model.m[2].weight.data.copy_(state_dict['model.105.m.2.weight'].float())
    reference_model.m[2].bias.data.copy_(state_dict['model.105.m.2.bias'].float())
    reference_model.eval()
    torch_model.eval()

    torch.manual_seed(42)
    input_0 = torch.rand(1, 256, 64, 64) 
    input_1 = torch.rand(1, 512, 32, 32)  
    input_2 = torch.rand(1, 1024, 16, 16) 
    with torch.inference_mode():
        out_0 = reference_model.m[0](input_0) 
        out_1 = reference_model.m[1](input_1)  
        out_2 = reference_model.m[2](input_2)  

        out_3 = torch_model.m[0](input_0) 
        out_4 = torch_model.m[1](input_1)  
        out_5 = torch_model.m[2](input_2) 
    
    pcc_value = comp_pcc(out_0, out_3)
    pcc_value1 = comp_pcc(out_1, out_4)
    pcc_value2 = comp_pcc(out_2, out_5)

    assert pcc_value[0], pcc_value[1]
    assert pcc_value1[0], pcc_value1[1]
    assert pcc_value2[0], pcc_value2[1]
    
    

