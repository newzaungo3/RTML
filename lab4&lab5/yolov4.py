#%%
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets, models, transforms
import numpy as np
import cv2 
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from util import *
from darknet import *

yolo_4_path = "/root/labs/cfg/yolov4.cfg"
weight_path = "/root/labs/weight/csdarknet53-omega_final.weights"
#%%
import torch
import torchvision
from torchvision import datasets, models, transforms
class myCocoDetection(datasets.CocoDetection):
    def __getitem__(self, index):
        from PIL import Image
        import os
        import os.path
        import numpy as np
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = np.array(img)

        category_ids = [obj['category_id'] for obj in target]
        bboxes = [obj['bbox'] for obj in target]

        import albumentations as A
        transform = A.Compose([
            A.SmallestMaxSize(256),
            A.CenterCrop(width=224, height=224),
            ], 
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        )

        # bboxes = [obj['bbox'] for obj in target]
        # category_ids = [obj['category_id'] for obj in target]
        transformed = transform(image=img, bboxes=bboxes, category_ids=category_ids)
        img = transformed['image']
        bboxes = torch.Tensor(transformed['bboxes'])
        cat_ids = torch.Tensor(transformed['category_ids'])
        labels, bboxes = self.__create_label(bboxes, cat_ids.type(torch.IntTensor))

        return img, labels, bboxes

    def __create_label(self, bboxes, class_inds):
        """
        Label assignment. For a single picture all GT box bboxes are assigned anchor.
        1、Select a bbox in order, convert its coordinates("xyxy") to "xywh"; and scale bbox'
           xywh by the strides.
        2、Calculate the iou between the each detection layer'anchors and the bbox in turn, and select the largest
            anchor to predict the bbox.If the ious of all detection layers are smaller than 0.3, select the largest
            of all detection layers' anchors to predict the bbox.
        Note :
        1、The same GT may be assigned to multiple anchors. And the anchors may be on the same or different layer.
        2、The total number of bboxes may be more than it is, because the same GT may be assigned to multiple layers
        of detection.
        """
        ANCHORS = [
            [[12, 16], [19, 36], [40, 28]],
            [[36, 75], [76, 55], [72, 146]],
            [[142, 110], [192, 243], [459, 401]]
        ]

        STRIDES = [8, 16, 32]

        IP_SIZE = 224
        NUM_ANCHORS = 3
        NUM_CLASSES = 80
        import json
        with open('/root/labs/coco_cats.json') as js:
            data = json.load(js)["categories"]

        cats_dict = {}
        for i in range(0, 80):
            cats_dict[str(data[i]['id'])] = i
        # print("Class indices: ", class_inds)
        bboxes = np.array(bboxes)
        class_inds = np.array(class_inds)
        anchors = ANCHORS # all the anchors
        strides = np.array(STRIDES) # list of strides
        train_output_size = IP_SIZE / strides # image with different scales
        anchors_per_scale = NUM_ANCHORS # anchor per scale

        # print(train_output_size)

        label = [
            np.zeros(
                (
                    int(train_output_size[i]),
                    int(train_output_size[i]),
                    anchors_per_scale,
                    5 + NUM_CLASSES,
                )
            )
            for i in range(3)
        ]
        # for i in range(3):
            # label[i][..., 5] = 1.0

        # 150 bounding box ground truths per scale
        bboxes_xywh = [
            np.zeros((150, 4)) for _ in range(3)
        ]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for i in range(len(bboxes)):
            bbox_coor = bboxes[i][:4]
            bbox_class_ind = cats_dict[str(class_inds[i])]
            # bbox_mix = bboxes[i][5]

            # onehot
            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            # one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate(
                [
                    (0.5 * bbox_coor[2:] + bbox_coor[:2]) ,
                    bbox_coor[2:],
                ],
                axis=-1,
            )
            # print("bbox_xywh: ", bbox_xywh)
            
            bbox_xywh_scaled = (
                1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
            )

            # print("bbox_xywhscaled: ", bbox_xywh_scaled)

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )  # 0.5 for compensation

                # assign all anchors 
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = iou_xywh_numpy(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    # Bug : 当多个bbox对应同一个anchor时，默认将该anchor分配给最后一个bbox
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh * strides[i]
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = one_hot

                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150为一个先验值,内存消耗大
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh * strides[i]
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                # check if a ground truth bb have the best anchor with any scale
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh * strides[best_detect]
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                # label[best_detect][yind, xind, best_anchor, 5:6] = bbox_mix
                label[best_detect][yind, xind, best_anchor, 5:] = one_hot 

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh * strides[best_detect]
                bbox_count[best_detect] += 1

        flatten_size_s = int(train_output_size[2]) * int(train_output_size[2]) * anchors_per_scale
        flatten_size_m = int(train_output_size[1]) * int(train_output_size[1]) * anchors_per_scale
        flatten_size_l = int(train_output_size[0]) * int(train_output_size[0]) * anchors_per_scale

        label_s = torch.Tensor(label[2]).view(1, flatten_size_s, 5 + NUM_CLASSES).squeeze(0)
        label_m = torch.Tensor(label[1]).view(1, flatten_size_m, 5 + NUM_CLASSES).squeeze(0)
        label_l = torch.Tensor(label[0]).view(1, flatten_size_l, 5 + NUM_CLASSES).squeeze(0)

        bboxes_s = torch.Tensor(bboxes_xywh[2])
        bboxes_m = torch.Tensor(bboxes_xywh[1])
        bboxes_l = torch.Tensor(bboxes_xywh[0])

        # label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        # print("label")
        labels = torch.cat([label_l, label_m, label_s], 0)
        bboxes = torch.cat([bboxes_l, bboxes_m, bboxes_s], 0)
        return labels, bboxes


def iou_xywh_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x,y,w,h)，其中(x,y)是bbox的中心坐标
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    # print(boxes1, boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

# %%
path2data="/root/labs/COCO/val2017"
path2json="/root/labs/COCO/annotations/instances_val2017.json"

mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

preprocess_augment = transforms.Compose([
    transforms.Resize(608),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


full_dataset = myCocoDetection(root=path2data, annFile=path2json)

BATCH_SIZE = 1
NUM_WORKERS = 0


train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [4000, 1000])

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True , num_workers=NUM_WORKERS)

val_dataloader  = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# # get some random training images
dataiter = iter(train_dataloader)
print(dataiter.next())

images, labels, boxes = dataiter.next()

toshow = torchvision.utils.make_grid(np.transpose(images, (0,3,1,2)))
toshow = toshow / 2 + 0.5     # unnormalize
npimg = toshow.numpy()

fig, ax = plt.subplots(figsize=(16,9))

ax.imshow(np.transpose(images, (0,1,2,3) )[0] )

print(boxes)
print(labels.shape)
ax.imshow(images)
print(labels[1]['bbox'])
dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
for index, img in enumerate(images):

    for i,l in enumerate(labels):
        x,y,w,h = labels[i]['bbox']
        x,y,w,h = float(x),float(y),float(w),float(h)
        # print(x,y,w,h)
        rect = patches.Rectangle(xy=(x,y),width=w,height=h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
plt.show()
print(labels, classes[labels])




# %%
# import torch
# from torchvision import datasets, models, transforms
from pycocotools.coco import COCO
root = "COCO/val2017"
annFile = 'COCO/annotations/instances_val2017.json'
coco = COCO(annFile)
ids = list(coco.imgs.keys())

from imgaug import augmenters as iaa 
import imgaug as ia
ia.seed(1)

seq = iaa.Sequential([
    iaa.Resize({"longer-side": 608, "shorter-side": "keep-aspect-ratio"}),
    iaa.CenterPadToSquare()
])

from PIL import Image
import os
import os.path
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
img_id = ids[0]
ann_ids = coco.getAnnIds(imgIds=img_id)
target = coco.loadAnns(ann_ids)
path = coco.loadImgs(img_id)[0]['file_name']
img = Image.open(os.path.join(root, path)).convert('RGB')
# t = transforms.Compose([transforms.ToTensor()])
# img = t(img)
img = np.array(img)
boxes = list()
for i in target:
    x,y,w,h = i['bbox']
    # print(x,y,w,h)
    x1,y1 = x,y
    x2,y2 = x+w,y+h

    boxes.append(BoundingBox(x1=x1,y1=y1,x2=x2,y2=y2))

print(img.shape)
boxes = BoundingBoxesOnImage(boxes, shape=img.shape)

img_aug, boxes_aug = seq(image=img, bounding_boxes=boxes)
# img = preprocess(img)
# target = target_transform(target)

ia.imshow(boxes.draw_on_image(img))

ia.imshow(boxes_aug.draw_on_image(img_aug))
# %%
