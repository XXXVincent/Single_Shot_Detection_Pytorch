import torch
import torch.nn as nn
import torch.functional as F
from utils import match

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes,iou_thresh, neg_pos, use_gpu=False):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = iou_thresh
        self.negpos_ratio = neg_pos
        self.variance = 0
    def forward(self, pred, targets):
        loc = pred['loc']
        conf = pred['conf']
        priorbox = pred['priorbox']
        batch_size = loc.size(0)
        num_priors = loc.size(1)

        loc_t = torch.Tensor(batch_size, num_priors, 4)
        conf_t = torch.LongTensor(batch_size, num_priors)

        for idx in range(batch_size):
            truths = targets[idx][:, :-1].detach()
            labels = targets[idx][:, -1].detach()
            defaults = priorbox.detach()
            match(self.threshold, truths, defaults, self.variance,
                  labels, loc_t, conf_t, idx)

        pos = conf_t > 0
        pos_idx = pos.unsqueeze(2).expand_as(loc)
        loc_p = loc[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t)

        batch_conf = conf.view(-1, self.num_classes)
        conf_logP = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        conf_logP = conf_logP.view(batch_size, -1)
        conf_logP[pos] = 0
        _, index = conf_logP.sort(1, descending=True)
        _, idx_rank = indel.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg

        pos_idx = pos.unsqueeze(2).expand_as(conf)
        neg_idx = neg.unsqueeze(2).expand_as(conf)

        conf_p = conf[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        conf_target = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, conf_target)

        N = num_pos.detach().sum().float()

        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

if __name__ == "__main__":
    loss = MultiBoxLoss(21, 0.5, 3)
    p = (torch.randn(1, 100, 4), torch.randn(1, 100, 21), torch.randn(100, 4))
    t = torch.randn(1, 10, 4)
    tt = torch.randint(20, (1, 10, 1))
    t = torch.cat((t, tt.float()), dim=2)
    p = {}
    p['loc'] = torch.randn(1, 100, 4)
    p['conf'] = torch.randn(1, 100, 21)
    p['priorbox'] = torch.randn(100, 4)
    l, c = loss(p, t)
    # 随机randn,会导致g_wh出现负数，此时结果会变成 nan
    print('loc loss:', l)
    print('conf loss:', c)