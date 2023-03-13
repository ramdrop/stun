#%%
import sys

sys.path.append('..')
from re import L
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import cirtorch.functional as LF
import math
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Backbone(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.sigma_dim = 2048
        self.mu_dim = 2048

        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, verbose=False)
        features = list(resnet50.children())[:-2]
        # feature map: ([B,3,224,224])->([B,2048,7,7])

        self.backbone = nn.Sequential(*features, GeM(), nn.Flatten())
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)


class TeacherNet(Backbone):
    def __init__(self, opt=None):
        super().__init__()
        self.id = 'teacher'
        self.mean_head = nn.Sequential(L2Norm(dim=1))

    def forward(self, inputs):
        B, C, H, W = inputs.shape                # (B, 1, 3, 224, 224)
                                                 # inputs = inputs.view(B * L, C, H, W)     # ([B, 3, 224, 224])

        backbone_output = self.backbone(inputs)                                                    # ([B, 2048, 1, 1])
        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])

        return mu, torch.zeros_like(mu)


class StudentNet(TeacherNet):
    def __init__(self, opt=None):
        super().__init__()
        self.id = 'student'
        self.var_head = nn.Sequential(nn.Linear(2048, self.sigma_dim), nn.Sigmoid())

    def forward(self, inputs):
        B, C, H, W = inputs.shape                # (B, 1, 3, 224, 224)
        inputs = inputs.view(B, C, H, W)         # ([B, 3, 224, 224])

        backbone_output = self.backbone(inputs)                                                    # ([B, 2048, 7, 7])
        mu = self.mean_head(backbone_output).view(B, -1)                                           # ([B, 2048]) <= ([B, 2048, 1, 1])
        log_sigma_sq = self.var_head(backbone_output).view(B, -1)                                  # ([B, 2048]) <= ([B, 2048, 1, 1])

        return mu, log_sigma_sq


def deliver_model(opt, id):
    if id == 'tea':
        return TeacherNet(opt)
    elif id == 'stu':
        return StudentNet(opt)


if __name__ == '__main__':
    tea = TeacherNet()
    stu = StudentNet()
    inputs = torch.rand((1, 3, 224, 224))
    outputs_tea = tea(inputs)
    outputs_stu = stu(inputs)

    print(outputs_tea[0].shape, outputs_tea[1].shape)
    print(outputs_tea[0].shape, outputs_stu[1].shape)
