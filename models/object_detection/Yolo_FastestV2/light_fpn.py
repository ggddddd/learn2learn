import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvblock(nn.Module):

    def __init__(self, input_channels, output_channels, size):
        super(DWConvblock, self).__init__()
        self.size = size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LightFPN(nn.Module):

    def __init__(self, stage_in_channels, out_depth):
        super(LightFPN, self).__init__()

        self.inputs_num = len(stage_in_channels)

        stage_in_channels = [sum(stage_in_channels[i:]) for i in range(self.inputs_num)]

        self.stages = nn.ModuleList()

        for channels in stage_in_channels:
            conv1x1 = nn.Sequential(nn.Conv2d(channels, out_depth, 1, 1, 0, bias=False), nn.BatchNorm2d(out_depth),
                                    nn.ReLU(inplace=True))
            cla_head = DWConvblock(channels, out_depth, 5)
            reg_head = DWConvblock(channels, out_depth, 5)

            self.stages.append(nn.ModuleList([conv1x1, cla_head, reg_head]))

    def forward(self, inputs):
        assert len(inputs) == self.inputs_num, f"except {self.inputs_num} feature maps, got {len(inputs)}"
        outputs = []
        last_feature = None
        for feature, (conv_1x1, cla_head, reg_head) in zip(inputs[::-1], self.stages[::-1]):  # 自底向上
            if last_feature is not None:
                last_feature_resized = F.interpolate(last_feature, scale_factor=2)
                last_feature = feature
                feature = torch.cat((last_feature_resized, feature), 1)
            else:
                last_feature = feature

            x = conv_1x1(feature)
            obj = cla = cla_head(x)
            reg = reg_head(x)

            outputs.append([cla, obj, reg])

        return outputs[::-1]


if __name__ == '__main__':
    model = LightFPN([96, 192], 72)
    c1 = torch.rand(1, 96, 20, 20)
    c2 = torch.rand(1, 192, 10, 10)
    test_outputs = model([c1, c2])
    for out in test_outputs:
        for item in out:
            print(item.size())
