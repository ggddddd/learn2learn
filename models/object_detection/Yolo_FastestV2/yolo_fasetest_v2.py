import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

from light_fpn import LightFPN
from models.backbones.shufflenetv2 import ShuffleNetV2


class Detector(nn.Module):

    def __init__(self, classes, anchor_num, load_param, export_onnx=False):
        super(Detector, self).__init__()
        out_depth = 72
        stage_out_channels = [24, 48, 96, 192]
        stage_repeats = [-1, 4, 8, 4]
        out_indices = (2, 3)

        self.export_onnx = export_onnx
        self.backbone = ShuffleNetV2(stage_out_channels, stage_repeats, load_param, out_indices)
        out_channels = [stage_out_channels[i] for i in out_indices]
        self.fpn = LightFPN(out_channels, out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cla_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

    def forward(self, x):
        features = self.backbone(x)
        features = self.fpn(features)
        
        outputs = []
        for cla, obj, reg in features:
            out_reg = self.output_reg_layers(reg)
            out_obj = self.output_obj_layers(obj)
            out_cla = self.output_cla_layers(cla)

            if self.export_onnx:
                out_reg = out_reg.sigmoid()
                out_obj = out_obj.sigmoid()
                out_cla = F.softmax(out_cla, dim=1)
                outputs.append(torch.cat((out_reg, out_obj, out_cla), 1).permute(0, 2, 3, 1))
            else:
                outputs.append([out_reg, out_obj, out_cla])
        return outputs


if __name__ == "__main__":
    model = Detector(80, 3, True)
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(
        model,  #model being run
        test_data,  # model input (or a tuple for multiple inputs)
        "test.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True)  # whether to execute constant folding for optimization
