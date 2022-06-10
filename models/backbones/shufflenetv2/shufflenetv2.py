import torch
import torch.nn as nn


class ShuffleV2Block(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.numel()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):

    def __init__(self, stage_out_channels, stage_repeats, load_param=False, out_indices=(0, 1, 2)):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = stage_repeats[1:]
        self.stage_out_channels = stage_out_channels

        # 需要输出的特征
        self.out_indices = out_indices

        # building first layer
        input_channel = self.stage_out_channels[0]
        first_conv = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(input_channel),
                                   nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.stages = nn.ModuleList([first_conv])
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 1]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(
                        ShuffleV2Block(input_channel,
                                       output_channel,
                                       mid_channels=output_channel // 2,
                                       ksize=3,
                                       stride=2))
                else:
                    stageSeq.append(
                        ShuffleV2Block(input_channel // 2,
                                       output_channel,
                                       mid_channels=output_channel // 2,
                                       ksize=3,
                                       stride=1))
                input_channel = output_channel
            self.stages.append(nn.Sequential(*stageSeq))

        if load_param == False:
            self.init_weights()

    def init_weights(self):
        """
        Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):

        outputs = []

        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                outputs.append(x)

        return outputs


if __name__ == "__main__":
    model = ShuffleNetV2([24, 48, 96, 192], stage_repeats=[-1, 4, 8, 4], out_indices=(0, 1, 2, 3, 4))
    test_data = torch.rand(1, 3, 320, 320)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())
