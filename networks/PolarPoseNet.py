import torch
import torch.nn as nn
from .resnet_backbone import ResNetBackboneNet, resnet_spec
from .fusion import LateFusion
from .decoder import Decoder
from .posehead import ConvPnPNet
from mmcv.runner import load_checkpoint

class PolarPoseNet(nn.Module):
    def __init__(self, fusion, backbone0, backbone1, decoder, pose_head_net):
        super(PolarPoseNet, self).__init__()

        self.fusion = fusion
        self.encoder0 = backbone0
        self.encoder1 = backbone1
        self.decoder = decoder
        self.pose_head_net = pose_head_net

    def forward(self, inputs):

        fused_rgb, fused_normal = self.fusion(inputs)
        features_0 = self.encoder0(fused_rgb)
        features_1 = self.encoder1(fused_normal)
        mask, normals, coor_x, coor_y, coor_z = self.decoder(features_0, features_1)
        nocs = torch.cat([coor_x, coor_y, coor_z], dim=1)
        coor_feat = torch.cat([nocs, inputs["roi_coord_2d"]], dim=1)
        rot, trans = self.pose_head_net(coor_feat, normals, inputs["roi_extents"])


        return mask, normals, nocs, rot, trans


def build_model(cfg):

    fusion_net = LateFusion()
    block_type, layers, channels, name = resnet_spec[cfg.num_layers]
    encoder_backbone_0 = ResNetBackboneNet(block_type, layers, in_channel=3, use_skips=True)
    encoder_backbone_1 = ResNetBackboneNet(block_type, layers, in_channel=3, use_skips=True)
    decoder = Decoder(num_encoder_layers=cfg.num_layers, num_decoder_layers=3, mask_output_dim=1, normal_output_dim=3, nocs_output_dim=3)
    pose_head = ConvPnPNet()

    model = PolarPoseNet(fusion=fusion_net, backbone0=encoder_backbone_0, backbone1=encoder_backbone_1, decoder=decoder,pose_head_net=pose_head)

    if cfg.pretrained:
        backbone_pretrained = cfg.pretrained_backbone
        print("load backbone weights from: {}".format(backbone_pretrained))
        load_checkpoint(model.encoder0, backbone_pretrained, strict=False)
        load_checkpoint(model.encoder1, backbone_pretrained, strict=False)
    else:
        print("Randomly initialize weights for backbone!")

    model.to(torch.device(cfg.device))

    return model

