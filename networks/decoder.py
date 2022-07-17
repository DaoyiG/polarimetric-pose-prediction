import torch
import torch.nn as nn
import torch.nn.functional
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from .resnet_backbone import resnet_spec


class Decoder(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, mask_output_dim, normal_output_dim, nocs_output_dim):
        super(Decoder, self).__init__()

        self.mask_output_dim = mask_output_dim
        self.normal_output_dim = normal_output_dim
        self.nocs_output_dim = nocs_output_dim
        _, _, channels, _ = resnet_spec[num_encoder_layers]

        self.features = nn.ModuleList()
        self.features.append(
            nn.ConvTranspose2d(
                512 + 512,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )
        )
        self.features.append(nn.BatchNorm2d(256))
        self.features.append(nn.ReLU(inplace=True))
        for i in range(num_decoder_layers):
            if i < 2:
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(
                        256 + channels[-2 - i] + channels[-2 - i], 256, kernel_size=3, stride=1, padding=1,
                        bias=False
                    )
                )

            else:
                self.features.append(
                    nn.Conv2d(
                        256, 256, kernel_size=3, stride=1, padding=1, bias=False
                    )
                )

            self.features.append(nn.BatchNorm2d(256))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.features.append(nn.BatchNorm2d(256))
            self.features.append(nn.ReLU(inplace=True))

        self.features.append(
            nn.Conv2d(
                256,
                self.mask_output_dim + self.normal_output_dim + self.nocs_output_dim,  # 1+3+3
                kernel_size=1,
                padding=0,
                bias=True,
            )
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def forward(self, features0, features1):

        x = torch.cat([features0[-1], features1[-1]], dim=1)
        for i, layer in enumerate(self.features):
            if i == 3:
                x = torch.cat([x, features0[-2], features1[-2]], 1)
            elif i == 10:
                x = torch.cat([x, features0[-3], features1[-3]], 1)

            x = layer(x)

        mask = x[:, : self.mask_output_dim, :, :]  #
        nocs = x[:, self.mask_output_dim: self.mask_output_dim + self.nocs_output_dim, :, :]
        normals = x[:, self.mask_output_dim + self.nocs_output_dim:, :, :]
        bs, c, h, w = nocs.shape
        nocs = nocs.view(bs, 3, self.nocs_output_dim // 3, h, w)
        coor_x = nocs[:, 0, :, :, :]
        coor_y = nocs[:, 1, :, :, :]
        coor_z = nocs[:, 2, :, :, :]

        return mask, normals, coor_x, coor_y, coor_z
