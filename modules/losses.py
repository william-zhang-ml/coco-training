""" Various composite loss functions for single stage object detection. """
from typing import Tuple
import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class DetectionBMCLoss:
    """ Composite object detection criteria for Yolo-style targets. """
    def __init__(self,
                 weight: Tuple[float, float, float] = (1, 1, 1)) -> None:
        """Bookkeep loss hyperparameters.

        Args:
            weight (Tuple[float, float float]): scaling coefficients ->
                [0] detection, [1] regression, [2] classification
        """
        self.weight = weight
        self.det_criteria = nn.BCEWithLogitsLoss(reduction='none')
        self.regr_criteria = nn.MSELoss(reduction='none')
        self.cls_criteria = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple:

        # this var is needed to correctly average object-specific target vals
        num_obj = target[:, 0].sum(dim=[1, 2, 3])
        num_bg = target[0][0].numel() - num_obj
        num_obj, num_bg = num_obj + 1e-9, num_bg + 1e-9

        # compute object detection / background rejection loss
        det_loss = self.det_criteria(pred[:, 0], target[:, 0])
        det_loss = [
            (det_loss * target[:, 0]).sum(dim=[1, 2, 3]) / num_obj,
            (det_loss * (1 - target[:, 0])).sum(dim=[1, 2, 3]) / num_bg
        ]
        det_loss = (det_loss[0] + det_loss[1]) / 2
        det_loss = det_loss * self.weight[0]

        # compute regression mean-squared-error (by object)
        regr_loss = self.regr_criteria(pred[:, 1:5], target[:, 1:5])
        regr_loss = regr_loss.sum(dim=1) / 4
        regr_loss = regr_loss * target[:, 0]
        regr_loss = regr_loss.sum(dim=[1, 2, 3]) / num_obj
        regr_loss = regr_loss * self.weight[1]

        # compute mean multiclass-crossentropy (by object)
        cls_loss = self.cls_criteria(pred[:, 5:], target[:, 5].long())
        cls_loss = cls_loss * target[:, 0]
        cls_loss = cls_loss.sum(dim=[1, 2, 3]) / num_obj
        cls_loss = cls_loss * self.weight[2]

        return det_loss, regr_loss, cls_loss

    def __repr__(self) -> str:
        argstr = f'weight={self.weight}'
        return f'DetectionBMCLoss({argstr})'


# pylint: disable = too-few-public-methods
class DetectionFMCLoss(DetectionBMCLoss):
    """ Composite object detection criteria for Yolo-style targets. """
    def __init__(self,
                 weight: Tuple[float, float, float] = (1, 1, 1)) -> None:
        """Bookkeep loss hyperparameters.

        Args:
            weight (Tuple[float, float float]): scaling coefficients ->
                [0] detection, [1] regression, [2] classification
        """
        super().__init__(weight)
        self.det_criteria = lambda pred, targ: sigmoid_focal_loss(
            pred, targ, alpha=0.5, reduction='none')

    def __repr__(self) -> str:
        argstr = f'weight={self.weight}'
        return f'DetectionFMCLoss({argstr})'
# pylint: enable = too-few-public-methods
