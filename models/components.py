
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss_zoo import cross_entropy




class FC(nn.Module):
    def __init__(self, dim_feature, num_classes):
        super(FC, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(dim_feature, num_classes)
        
        # init the fc layer
        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, dim_feature):
        return self.fc(dim_feature)

class Wrapper(nn.Module):

    def __init__(self, arch, fc):
        super(Wrapper, self).__init__()
        self.arch = arch
        self.fc = fc

    def forward(self, inputs):
        features = self.arch(inputs)
        outs = self.fc(features)
        return outs

class ISDAWrapper(Wrapper):

    def __init__(self, arch, fc, leverage=0.5, epochs=200) :
        super(ISDAWrapper, self).__init__(arch, fc)

        self.leverage = leverage / 2
        self.epochs = epochs
        self.dim_feature = self.arch.dim_feature
        self.num_classses = self.fc.num_classes
        self.register_buffer("cov", torch.zeros(self.dim_feature, self.dim_feature, self.num_classses))
        self.register_buffer("means", torch.zeros(self.dim_feature, self.num_classses))
        self.register_buffer("nums", torch.zeros(self.num_classses))

    def _update_cov(self, features, labels):
        features = features.detach()
        counts = features.size(0)
        features = features.view(
            counts, self.dim_feature, 1
        ).expand(counts, self.dim_feature, self.num_classses) # N x D x C 
        one_hots = F.one_hot(labels, self.num_classses).view(
            counts, 1, self.num_classses
        ).expand(counts, self.dim_feature, self.num_classses) # N x D x C
        features_aligned = features * one_hots
        nums = one_hots.sum(dim=0)[0] # C
        nums[nums == 0] = 1 # avoid zero
        means = features_aligned.sum(dim=0) / nums # D x C
        features_aligned = (features_aligned - means) * one_hots
        cov = torch.einsum(
            "nic,njc->ijc",
            features_aligned,
            features_aligned
        ) / nums

        # start to update the cov, means and nums
        part1 = (self.nums * self.cov + nums * cov) / (self.nums + nums)
        tmp = self.means - means
        part2 = self.nums * nums * torch.einsum(
            "ic,jc->ijc",
            tmp,
            tmp
        ) / (self.nums + nums).pow(2)
        self.cov = part1 + part2
        self.means = (self.nums * self.means + nums * means) / (self.nums + nums)
        self.nums = self.nums + nums

    def _loss(self, outs, labels, cur_epoch):
        n = labels.size(0)
        weights = self.fc.fc.weight.data.clone().detach().T # D x C
        weights_extend = weights.unsqueeze(-1).repeat(1, 1, n) # D x C x N
        weights = weights[:, labels] # D x N
        weights_tmp = weights_extend - weights.unsqueeze(1)
        covs = self.cov[:, :, labels]
        part2 = torch.einsum(
            "icn,ijn,jcn->nc",
            weights_tmp, covs, weights_tmp
        ) # N x C
        leverage = self.leverage * cur_epoch / self.epochs
        outs = outs + part2 * leverage
        loss = cross_entropy(outs, labels, reduction="mean")
        return loss
        
    def forward(self, inputs, labels=None, cur_epoch=None):
        features = self.arch(inputs)
        outs = self.fc(features)
        if self.training:
            self._update_cov(features, labels)
            return outs, self._loss(outs, labels, cur_epoch)
        else:
            return outs

        
