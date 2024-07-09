import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numba import jit


# 将Numpy数组转换为PyTorch张量，并在需要时将其放置到GPU上
def to_tensor(array, requires_grad=False, device='cpu', dtype=torch.float32):
    return Variable(torch.tensor(array, requires_grad=requires_grad, device=device, dtype=dtype))

# 自定义损失函数，结合交叉熵损失和距离图损失
class CustomLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_distance=1.0, distance_weight=1.0, distance_constraint=None):
        super(CustomLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_distance = weight_distance
        self.distance_weight = distance_weight
        self.distance_constraint = distance_constraint


    def forward(self, output, target, distance_map):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(output, target)

        # 计算距离范围约束项
        constraint_loss = torch.clamp(distance_map - self.distance_constraint[0], min=0.0) + \
                          torch.clamp(self.distance_constraint[1] - distance_map, min=0.0)

        # 综合两个损失
        total_loss = self.weight_ce * ce_loss + constraint_loss.mean()

        return total_loss


