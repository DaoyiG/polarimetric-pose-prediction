import torch
import torch.nn as nn
from mmcv.cnn import normal_init, constant_init

class LateFusion(nn.Module):
    def __init__(self):
        super(LateFusion, self).__init__()

        self.conv_rgb_naive = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv_normals_naive = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv_normals = nn.Sequential(
            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv_aolp_naive = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv_dolp_naive = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.conv_rgb_aolp_dolp_fuse = nn.Sequential(
            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, inputs):

        rgb_0_conv = self.conv_rgb_naive(inputs["roi_img_0"])
        rgb_45_conv = self.conv_rgb_naive(inputs["roi_img_45"])
        rgb_90_conv = self.conv_rgb_naive(inputs["roi_img_90"])
        rgb_135_conv = self.conv_rgb_naive(inputs["roi_img_135"])

        rgb = torch.cat((rgb_0_conv, rgb_45_conv, rgb_90_conv, rgb_135_conv), dim=1)
        rgb = self.conv_rgb(rgb)

        aolp_conv = self.conv_aolp_naive(inputs["roi_aolp"])
        dolp_conv = self.conv_dolp_naive(inputs["roi_dolp"])

        rgb = self.conv_rgb_aolp_dolp_fuse(torch.cat((rgb, aolp_conv, dolp_conv), dim=1))

        diff_normals_conv = self.conv_normals_naive(inputs["roi_N_diff"])
        spec1_normals_conv = self.conv_normals_naive(inputs["roi_N_spec1"])
        spec2_normals_conv = self.conv_normals_naive(inputs["roi_N_spec2"])
        normals = self.conv_normals(torch.cat((diff_normals_conv, spec1_normals_conv, spec2_normals_conv), dim=1))


        return rgb, normals

