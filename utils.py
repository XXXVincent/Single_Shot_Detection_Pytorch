import torch



def match(*args):
    raise NotImplementedError

def iou(box_a, box_b):
    N = box_a.size(0)
    M = box_b.size(0)

    LT = torch.max(
        box_a[:, :2].unsqueeze(1).expand(N, M, 2),
        box_b[:, :2].unsuqeeze(0).expand(N, M, 2)
    )

    RB = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(N, M, 2),
        box_b[:, 2:].unsuqeeze(0).expand(N, M, 2)
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
