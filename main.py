"""
Train one-stage object detection model on COCO 2017.

Author: William Zhang
"""
import os
import shutil
import sys
from typing import List, Tuple
import yaml
from PIL.ImageDraw import Draw
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.ops import box_convert
from tqdm import tqdm
from coco import CocoInstances
from components.blocks import ConvBlock
from components.detectors import ConvDetector
from modules import labels, losses, utils


def collate(items: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate CocoInstances samples.

    Args:
        items (List[Tuple]): CocoInstances images and annotations

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: batch of images, target tensor

    Individual items expected to be torch.Tensor and dict.
    Outputs are shape (B, C, H, W) and (B, C', A, H', W').
    """

    # transform and collate images
    trans = transforms.Compose([
        transforms.Normalize(mean=128, std=128),
        transforms.Resize(CONFIG['image_size'])
    ])
    images = torch.stack([trans(itm[0].float()) for itm in items])

    # scale (wrt to above resize) and collate bounding boxes
    batch_idx, boxes = [], []
    for i_samp, annot in enumerate(itm[1] for itm in items):
        x_scale = CONFIG['image_size'][1] / items[i_samp][0].shape[-1]
        y_scale = CONFIG['image_size'][0] / items[i_samp][0].shape[-2]
        for instance in annot:
            batch_idx.append(i_samp)
            boxes.append([
                instance['bbox'][0] * x_scale,  # xywh
                instance['bbox'][1] * y_scale,
                instance['bbox'][2] * x_scale,
                instance['bbox'][3] * y_scale,
                instance['category_id'] - 1  # come as 1-indexed
            ])
    batch_idx, boxes = torch.tensor(batch_idx), torch.tensor(boxes)
    boxes[:, :4] = box_convert(boxes[:, :4], 'xywh', 'cxcywh')

    # generate target vectors and populate target tensor
    anchor_idx, row, col, target_vecs = labels.get_target_vecs(
        boxes=boxes,
        anchors=torch.tensor(CONFIG['anchors']),
        targ_xres=CONFIG['targ_res'][1],
        targ_yres=CONFIG['targ_res'][0],
    )
    target = torch.zeros(
        len(images),
        6,
        len(CONFIG['anchors']),
        CONFIG['image_size'][0] // CONFIG['targ_res'][0],
        CONFIG['image_size'][1] // CONFIG['targ_res'][1])
    target[batch_idx, :, anchor_idx, row, col] = target_vecs

    return images, target


def draw_overlay(filepath: str,
                 image: torch.Tensor,
                 box_tensor: torch.Tensor,
                 boxes_from_target: bool = False) -> None:
    """Overlay bounding boxes on image and save.

    Args:
        filepath (str): path to save final image to
        image (torch.Tensor): normalized image (C, H, W)
        box_tensor (torch.Tensor): target or logits tensor (B, C, A, H, W)
        boxes_from_target (bool, optional):
            boxes given as target vs logits flag. Defaults to False.
    """
    image = 128 * image + 128  # undo normalization
    image = transforms.ToPILImage()(image / 255)
    drawer = Draw(image)
    if boxes_from_target:
        boxes = labels.get_boxes_from_target(
            box_tensor.unsqueeze(0),
            torch.tensor(CONFIG['anchors']),
            targ_xres=CONFIG['targ_res'][1],
            targ_yres=CONFIG['targ_res'][0]
        )[0]
        boxes = boxes[:, :4]  # drop label column
    else:
        boxes = labels.get_boxes_from_logits(
            box_tensor.unsqueeze(0),
            torch.tensor(CONFIG['anchors']),
            targ_xres=CONFIG['targ_res'][1],
            targ_yres=CONFIG['targ_res'][0],
            det_thr=0.7,
            nms_thr=0.3
        )[0]
        boxes = boxes[:, 1:5]  # drop detection score and label columns
    for box in box_convert(boxes, 'cxcywh', 'xyxy').tolist():
        drawer.rectangle(box, outline='red')
    image.save(filepath)


def train(batches: torch.utils.data.DataLoader,
          model: nn.Module) -> None:
    """Train model.

    Args:
        batches (torch.utils.data.DataLoader): training batch generator
        model (nn.Module): model to train
    """
    criteria = getattr(losses, CONFIG['loss_class'])(**CONFIG['loss_kwargs'])
    optimizer = getattr(optim, CONFIG['optimizer_class'])(
        model.parameters(),
        **CONFIG['optimizer_kwargs']
    )
    if 'scheduler_class' in CONFIG:
        scheduler = getattr(optim.lr_scheduler, CONFIG['scheduler_class'])(
            optimizer,
            **CONFIG['scheduler_kwargs']
        )

    # sanity check box locations prior to training
    images, target = next(iter(batches))
    draw_overlay(
        os.path.join(OUT_DIR, 'sanity', 'target.png'),
        images[0],
        target[0],
        True
    )

    # main training loop
    prog_bar = tqdm(range(CONFIG['num_epochs']))
    for i_epoch in prog_bar:
        for i_batch, (images, target) in enumerate(batches):
            # forward pass
            logits = model(images)
            det_loss, regr_loss, cls_loss = criteria(logits, target)
            det_loss = det_loss.mean()
            regr_loss = regr_loss.mean()
            cls_loss = cls_loss.mean()
            total_loss = det_loss + regr_loss + cls_loss

            # backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # give user feedback and add to log
            prog_bar.set_postfix({
                'batch': f'{i_batch + 1}/{len(batches)}',
                'det_loss': f'{det_loss.detach().cpu().item():.03f}',
                'regr_loss': f'{regr_loss.detach().cpu().item():.03f}',
                'cls_loss': f'{cls_loss.detach().cpu().item():.03f}',
                'total_loss': f'{total_loss.detach().cpu().item():.03f}'
            })
            break  # overfit to 1 batch for now

        # checkpointing
        if (i_epoch + 1) % CONFIG['checkpoint_epochs'] == 0:
            draw_overlay(
                os.path.join(OUT_DIR, 'sanity', f'{i_epoch + 1:03d}.png'),
                images[0],
                logits[0]
            )
            torch.save(
                model.state_dict(),
                os.path.join(OUT_DIR, 'checkpoints', f'{i_epoch + 1:03d}.pt')
            )

        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    # get run name from user's CLI input and load config
    try:
        temp = os.path.join('configs', f'{ sys.argv[1]}.yaml')
        with open(temp, 'r', encoding='utf-8') as file:
            CONFIG = yaml.safe_load(file)
    except IndexError as e:
        raise IndexError('config file not provided in CLI') from e
    except FileNotFoundError as e:
        raise FileNotFoundError('could not find specified config file') from e

    # set up output directories
    OUT_DIR = os.path.join('runs', sys.argv[1])
    if os.path.isdir(OUT_DIR):
        temp = input(f'Run {sys.argv[1]} exists. Overwrite? ')
        while temp not in ['y', 'n']:
            temp = input(f'Run {sys.argv[1]} exists. Overwrite? ')
        if temp == 'y':
            shutil.rmtree(OUT_DIR)
        else:
            sys.exit()
    os.makedirs(os.path.join(OUT_DIR, 'checkpoints'))
    os.makedirs(os.path.join(OUT_DIR, 'sanity'))
    temp = os.path.join(OUT_DIR, f'{ sys.argv[1]}.yaml')
    with open(temp, 'w', encoding='utf-8') as file:
        yaml.dump(CONFIG, file)
    CONFIG = utils.flatten_config(CONFIG)

    # set up batch generator
    data_train = torch.utils.data.DataLoader(
        CocoInstances(
            metadata_path=CONFIG['metadata_path'],
            img_dir=CONFIG['image_dir']
        ),
        batch_size=CONFIG['batch_size'],
        collate_fn=collate,
        shuffle=False  # overfit to 1 batch for now
    )

    # train model
    my_model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        ConvBlock(32, 32, 3, 1, 1, groups=8),
        ConvBlock(32, 32, 3, 1, 1, groups=8),
        nn.MaxPool2d(2, 2),
        ConvBlock(32, 64, 3, 1, 1, groups=8),
        ConvBlock(64, 64, 3, 1, 1, groups=8),
        nn.MaxPool2d(2, 2),
        ConvBlock(64, 96, 3, 1, 1, groups=8),
        ConvBlock(96, 96, 3, 1, 1, groups=8),
        nn.MaxPool2d(2, 2),
        ConvBlock(96, 96, 3, 1, 1, groups=8),
        ConvDetector(96, 90, torch.tensor(CONFIG['anchors']))
    )
    train(data_train, my_model)
