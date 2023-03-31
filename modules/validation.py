""" Dataset-level human-interpretable metrics. """
from typing import Dict, List
import pandas as pd
import torch
from torchvision.ops import box_area, box_iou


def collate_iou(pred: List[torch.Tensor],
                truth: List[torch.Tensor]) -> torch.Tensor:
    """Collate truth box's max IOU with a network prediction.

    Args:
        pred (List[torch.Tensor]): per-sample network xyxy boxes (?, 4)
        truth (List[torch.Tensor]): per-sample truth xyxy boxes (?, 4)

    Returns:
        torch.Tensor: collated max IOU
    """
    agg_iou = []
    for curr_pred, curr_truth in zip(pred, truth):
        if len(curr_pred) == 0:
            agg_iou.append(torch.zeros(len(curr_truth)))
        elif len(curr_truth) > 0:
            agg_iou.append(box_iou(curr_truth, curr_pred).max(dim=1).values)
    return torch.cat(agg_iou)


def collate_area(boxes: List[torch.Tensor]) -> torch.Tensor:
    """Collate box area.

    Args:
        boxes (List[torch.Tensor]): xyxy boxes (?, 4)
    Returns:
        torch.Tensor: collated box area
    """
    agg_area = []
    for curr_boxes in boxes:
        if len(curr_boxes) > 0:
            agg_area.append(box_area(curr_boxes))
    return torch.cat(agg_area)


def recall_vs_size(pred: List[torch.Tensor],
                   truth: List[torch.Tensor],
                   sizes: List[float],
                   iou: float = 0.5) -> Dict[int, float]:
    """Compute recall based on object size (pixels-squared) bins.

    Args:
        pred (List[torch.Tensor]): per-sample network xyxy boxes (?, 4)
        truth (List[torch.Tensor]): per-sample truth xyxy boxes (?, 4)
        sizes (List[float]): ordered size bin boundaries (pixels-squared)
        iou (float, optional): minimum IOU to count as a detection.

    Returns:
        Dict[int, float]: map from size bin to recall
    """
    agg_iou = collate_iou(pred, truth)
    detected = agg_iou > iou
    agg_area = collate_area(truth)
    assert len(agg_iou) == len(agg_area)
    size_bin = torch.bucketize(agg_area, torch.tensor(sizes))
    table = pd.DataFrame({
        'detected': detected.numpy(),
        'size_bin': size_bin.numpy()
    })
    return table.groupby('size_bin').detected.mean().to_dict()
