""" Utility functions for making target vectors and decoding activations. """
from typing import List, Tuple
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert, box_iou, nms


def match_boxes(to_match: torch.Tensor, fits: torch.Tensor) -> Tuple:
    """Match bounding boxes to best-fit candidates.

    Args:
        to_match (torch.Tensor): bounding boxes to fit width and height (M, 2)
        fits (torch.Tensor): best-fit candidates width and height (N, 2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            IOU with best-fit (M, ), index of best-fit (M, )
    """
    to_match = torch.cat([torch.zeros_like(to_match), to_match], dim=1)
    to_match = box_convert(to_match, 'cxcywh', 'xyxy')
    fits = torch.cat([torch.zeros_like(fits), fits], dim=1)
    fits = box_convert(fits, 'cxcywh', 'xyxy')
    iou_max, iou_idx = box_iou(to_match, fits).max(dim=-1)
    return iou_max, iou_idx


def get_target_vecs(boxes: torch.Tensor,
                    anchors: torch.Tensor,
                    targ_xres: int,
                    targ_yres: int,
                    num_classes: int = None) -> Tuple:
    """Convert bounding boxes (cxcywh + label) to target vectors.

    Args:
        boxes (torch.Tensor): cxcywh + label bounding boxes (M, 5)
        anchors (torch.Tensor): anchor width and height
        targ_xres (int): number of input columns per target column
        targ_yres (int): number of input rows per target row
        num_classes (int): number of target classes

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            index of best-fit anchor,
            row location of target vector,
            column location of target vector,
            target vectors (row vectors)

    If num_classes is given, target vectors will have one-hot class labels.
    Otherwise the class label is kept as a categorical scalar.
    Target vectors will have shape (M, 5 + num_classes) or (M, 6).
    """
    col, row = boxes[:, 0] / targ_xres, boxes[:, 1] / targ_yres
    col, targ_x = col.trunc().long(), col.frac()
    row, targ_y = row.trunc().long(), row.frac()
    _, anchor_idx = match_boxes(boxes[:, 2:4], anchors)
    targ_w = (boxes[:, 2] / anchors[anchor_idx, 0]).log()
    targ_h = (boxes[:, 3] / anchors[anchor_idx, 1]).log()
    if num_classes is None:
        target_vecs = torch.cat(
            [
                torch.ones(len(boxes), 1),
                torch.stack([targ_x, targ_y, targ_w, targ_h], dim=1),
                boxes[:, 4:]
            ],
            dim=1
        )
    else:
        target_vecs = torch.cat(
            [
                torch.ones(len(boxes), 1),
                torch.stack([targ_x, targ_y, targ_w, targ_h], dim=1),
                F.one_hot(boxes[:, 4].long(), num_classes)
            ],
            dim=1
        )
    return anchor_idx, row, col, target_vecs


def get_boxes_from_target(target: torch.Tensor,
                          anchors: torch.Tensor,
                          targ_xres: int,
                          targ_yres: int,) -> List[torch.Tensor]:
    """Extract ground truth bounding box from target tensor.

    Args:
        target (torch.Tensor): target tensor (B, C, A, H, W)
        anchors (torch.Tensor): anchor width and height (N, )
        targ_xres (int): number of input columns per target column
        targ_yres (int): number of input rows per target row

    Returns:
        List[torch.Tensor]: per-sample bounding boxes
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(target.shape[-1]),
            torch.arange(target.shape[-2]),
            indexing='xy'
        )
    ).view(1, 2, 1, target.shape[-2], target.shape[-1])
    anchor_view = anchors.T.view(1, 2, -1, 1, 1)
    boxes = target[:, :6].clone()
    boxes[:, 1:3] += coords
    boxes[:, 1] *= targ_xres
    boxes[:, 2] *= targ_yres
    boxes[:, 3:5] = boxes[:, 3:5].exp() * anchor_view
    if boxes.shape[1] > 6:
        boxes[:, 5] = target[:, 5:].argmax(dim=1)
    boxes = boxes.view(boxes.shape[0], 6, -1)
    return [samp.T[samp[0] == 1, 1:] for samp in boxes]


# pylint: disable = too-many-arguments
def get_boxes_from_logits(logits: torch.Tensor,
                          anchors: torch.Tensor,
                          targ_xres: int,
                          targ_yres: int,
                          det_thr: float,
                          nms_thr: float = None) -> List[torch.Tensor]:
    """Extract ground truth bounding box from target tensor.

    Args:
        logits (torch.Tensor): detector output logits (B, C, A, H, W)
        anchors (torch.Tensor): anchor width and height (N, )
        targ_xres (int): number of input columns per target column
        targ_yres (int): number of input rows per target row
        det_thr: minimum acceptable detection score
        nms_thr: maximum IOU before suppressing a repeated detection

    Returns:
        List[torch.Tensor]: per-sample bounding boxes
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(logits.shape[-1]),
            torch.arange(logits.shape[-2]),
            indexing='xy'
        )
    ).view(1, 2, 1, logits.shape[-2], logits.shape[-1])
    anchor_view = anchors.T.view(1, 2, -1, 1, 1)
    boxes = logits[:, :6].clone()
    boxes[:, 0] = boxes[:, 0].sigmoid()
    boxes[:, 1:3] = boxes[:, 1:3].sigmoid() + coords
    boxes[:, 1] = boxes[:, 1] * targ_xres
    boxes[:, 2] = boxes[:, 2] * targ_yres
    boxes[:, 3:5] = boxes[:, 3:5].exp() * anchor_view
    boxes[:, 5] = logits[:, 5:].argmax(dim=1)
    boxes = boxes.view(boxes.shape[0], 6, -1)

    filtered = []
    for samp_boxes in boxes:
        samp_boxes = samp_boxes.T[samp_boxes[0] > det_thr]
        if nms_thr is not None:
            keep_mask = nms(
                box_convert(samp_boxes[:, 1:5], 'cxcywh', 'xyxy'),
                samp_boxes[:, 0],
                nms_thr
            )
            samp_boxes = samp_boxes[keep_mask]
        filtered.append(samp_boxes)
    return filtered
# pylint: enable = too-many-arguments
