import torch


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = iou(truths, point_from(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]

    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0

    loc = encode(matches, priors, variances)

    loc_t[idx] = loc
    conf_t[idx] = conf


def iou(box_a, box_b):
    N = box_a.size(0)
    M = box_b.size(0)

    LT = torch.max(
        box_a[:, :2].unsqueeze(1).expand(N, M, 2),
        box_b[:, :2].unsqueeze(0).expand(N, M, 2)
    )

    RB = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(N, M, 2),
        box_b[:, 2:].unsqueeze(0).expand(N, M, 2)
    )

    wh = RB - LT
    wh[wh < 0] = 0

    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # (N,)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])  # (M,)

    # 把面积的shape扩展为inter一样的（N，M）
    area_a = area_a.unsqueeze(1).expand_as(inter)
    area_b = area_b.unsqueeze(0).expand_as(inter)

    # iou
    iou = inter / (area_a + area_b - inter)

    return iou


def encode(matched, priors, variances):
    g_cxcy = (matched[:,:2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (priors[:, 2:] * variances[0])

    eps = 1e-5

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh + eps) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2]*variances[0]*priors[:, 2:],
                      priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes = point_from(boxes)
    return boxes

def point_from():
    raise NotImplementedError

if __name__ == "__main__":
    box_a = torch.Tensor([[2,1,4,3]])
    box_b = torch.Tensor([[3,2,5,4],
                          [3,2,5,4]])
    print('IOU = ',iou(box_a, box_b))