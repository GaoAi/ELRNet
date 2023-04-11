import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MyNetworks.layers import down_sampling_block, bt, non_bt, LFEM

class ELRNet(nn.Module):
    def __init__(self,
                 config,
                 down_factor=8,
                 interpolate=False,
                 dilation=True,
                 dropout=False,
                 ):
        super(ELRNet, self).__init__()

        self.name = 'ELRNet'
        self.nb_classes = config.nb_classes
        self.down_factor = down_factor
        self.interpolate = interpolate
        self.stage_channels = [-1, 16, 64, 128, 256, 512]

        if dilation == True:
            self.dilation_list = [1, 2, 4, 8, 16]
        else:
            self.dilation_list = [1, 1, 1, 1, 1 ]

        if dropout == True:
            self.dropout_list = [0.01, 0.001]

        if down_factor==8:
            # 8x downsampling
            self.encoder = nn.Sequential(
                down_sampling_block(3, 16),

                down_sampling_block(self.stage_channels[1], self.stage_channels[2]),
                LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),
                LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),
                LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),
                LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),
                LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2]),

                down_sampling_block(self.stage_channels[2], self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
                LFEM(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3]),
            )
            if interpolate == True:
                self.project_layer = nn.Conv2d(self.stage_channels[3], self.nb_classes, 1, bias=False)
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.stage_channels[3], self.stage_channels[2], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[2]),
                    nn.ReLU(inplace=True),

                    LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    LFEM(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2], ),
                    nn.ConvTranspose2d(self.stage_channels[2], self.stage_channels[1], kernel_size=3, stride=2,
                                       padding=1,
                                       output_padding=1, bias=False),
                    nn.BatchNorm2d(self.stage_channels[1]),
                    nn.ReLU(inplace=True),

                    LFEM(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1], ),
                    LFEM(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1], ),
                    nn.ConvTranspose2d(self.stage_channels[1], self.nb_classes, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=False),

                )


    def forward(self, input):

        encoder_out = self.encoder(input)

        if self.interpolate == True:
            decoder_out = self.project_layer(encoder_out)
            decoder_out = F.interpolate(decoder_out, scale_factor=self.down_factor, mode='bilinear', align_corners=True)
        else:
            decoder_out = self.decoder(encoder_out)

        return decoder_out

