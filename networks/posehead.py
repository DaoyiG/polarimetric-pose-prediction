import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init


class ConvPnPNet(nn.Module):
    def __init__(self, in_dim=8, rot_dim=6, trans_dim=3):
        super(ConvPnPNet, self).__init__()

        self.features = nn.ModuleList()
        for i in range(3):
            _in_channels = in_dim if i == 0 else 128
            self.features.append(nn.Conv2d(_in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False))
            self.features.append(nn.GroupNorm(num_groups=32, num_channels=128))
            self.features.append(nn.ReLU(inplace=True))

        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)

        self.fc_t = nn.Linear(256, trans_dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, coor_feat, normals, cur_extents):

        bs, in_c, fh, fw = coor_feat.shape
        if in_c == 3 or in_c == 5:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * cur_extents.view(bs, 3, 1, 1)
        x = torch.cat((coor_feat, normals), dim=1)


        for i, layer in enumerate(self.features):
            x = layer(x)

        x = x.view(-1, 128 * 8 * 8)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        rot = self.fc_r(x)
        t = self.fc_t(x)

        return rot, t

