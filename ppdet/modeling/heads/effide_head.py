# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

import math
import numpy as np
from ..initializer import bias_init_with_prob, constant_

from ..backbones.efficientrep import Conv
from ppdet.modeling.assigners.simota_assigner import SimOTAAssigner
from ppdet.modeling.bbox_utils import bbox_overlaps
from ..losses import IouLoss

from ppdet.modeling.layers import MultiClassNMS
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
import torch
import torchvision
from IPython import embed
### TODO

__all__ = ['EffiDeHead']


@register
@serializable
class EffiDeHead(nn.Layer):
    __shared__ = [
        'num_classes', 'width_mult', 'depth_mult', 'act', 'trt', 'exclude_nms'
    ]
    __inject__ = ['assigner', 'nms']

    def __init__(
            self,
            num_classes=80,
            depth_mult=1.0,
            width_mult=1.0,
            depthwise=False,
            in_channels=[128, 256, 512],
            fpn_strides=[8, 16, 32],
            anchors=1,
            num_layers=3,
            act='silu',
            assigner=SimOTAAssigner(use_vfl=False),
            nms='MultiClassNMS',
            loss_weight={
                'cls': 1.0,
                'obj': 1.0,
                'iou': 3.0,  # 5.0 
                'l1': 1.0,
                'reg': 5.0,
            },
            iou_type='ciou',
            trt=False,
            exclude_nms=False,
            head_layers=None):
        super().__init__()
        self._dtype = paddle.framework.get_default_dtype()
        self.num_classes = num_classes
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = [int(in_c * width_mult) for in_c in in_channels]
        feat_channels = self.in_channels
        self.fpn_strides = fpn_strides

        self.loss_weight = loss_weight
        self.iou_type = iou_type

        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms

        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [paddle.zeros([1])] * num_layers
        self.prior_prob = 1e-2
        self.stride = paddle.to_tensor(fpn_strides)

        ConvBlock = Conv
        self.stem_conv = nn.LayerList()
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()  # reg [x,y,w,h] + obj
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        for in_c, feat_c in zip(self.in_channels, feat_channels):
            self.stem_conv.append(Conv(in_c, feat_c, 1, 1))
            self.cls_convs.append(ConvBlock(feat_c, feat_c, 3, 1))
            self.reg_convs.append(ConvBlock(feat_c, feat_c, 3, 1))

            self.cls_preds.append(
                nn.Conv2D(
                    feat_c,
                    self.num_classes,
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            self.reg_preds.append(
                nn.Conv2D(
                    feat_c,
                    4,  # reg [x,y,w,h] + obj
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            self.obj_preds.append(
                nn.Conv2D(
                    feat_c,
                    1,  # reg [x,y,w,h] + obj
                    1,
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
        self.iou_loss = IouLoss(loss_weight=1.0)
        #self._initialize_biases()

    def _initialize_biases(self):
        bias_init = bias_init_with_prob(0.01)
        for cls_, obj_ in zip(self.cls_preds, self.obj_preds):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_init)
            constant_(obj_.weight)
            constant_(obj_.bias, bias_init)

    def _generate_anchor_point(self, feat_sizes, strides, offset=0.):
        anchor_points, stride_tensor = [], []
        num_anchors_list = []
        for feat_size, stride in zip(feat_sizes, strides):
            h, w = feat_size
            x = (paddle.arange(w) + offset) * stride
            y = (paddle.arange(h) + offset) * stride
            y, x = paddle.meshgrid(y, x)
            anchor_points.append(paddle.stack([x, y], axis=-1).reshape([-1, 2]))
            stride_tensor.append(
                paddle.full(
                    [len(anchor_points[-1]), 1], stride, dtype=self._dtype))
            num_anchors_list.append(len(anchor_points[-1]))
        anchor_points = paddle.concat(anchor_points).astype(self._dtype)
        anchor_points.stop_gradient = True
        stride_tensor = paddle.concat(stride_tensor)
        stride_tensor.stop_gradient = True
        return anchor_points, stride_tensor, num_anchors_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        feat_sizes = [[f.shape[-2], f.shape[-1]] for f in feats]
        cls_score_list, reg_pred_list = [], []
        obj_score_list = []
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            cls_logit = self.cls_preds[i](cls_feat)
            reg_xywh = self.reg_preds[i](reg_feat)
            obj_logit = self.obj_preds[i](reg_feat)
            # cls prediction
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            # reg prediction
            #reg_xywh, obj_logit = paddle.split(reg_pred, [4, 1], axis=1)
            reg_xywh = reg_xywh.flatten(2).transpose([0, 2, 1])
            reg_pred_list.append(reg_xywh)
            # obj prediction
            obj_score = F.sigmoid(obj_logit)
            obj_score_list.append(obj_score.flatten(2).transpose([0, 2, 1]))

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_pred_list = paddle.concat(reg_pred_list, axis=1)
        obj_score_list = paddle.concat(obj_score_list, axis=1)

        # bbox decode
        anchor_points, stride_tensor, _ =\
            self._generate_anchor_point(feat_sizes, self.fpn_strides)
        reg_xy, reg_wh = paddle.split(reg_pred_list, 2, axis=-1)
        reg_xy += (anchor_points / stride_tensor)
        reg_wh = paddle.exp(reg_wh) * 0.5
        bbox_pred_list = paddle.concat(
            [reg_xy - reg_wh, reg_xy + reg_wh], axis=-1)

        if self.training:
            anchor_points, stride_tensor, num_anchors_list =\
                self._generate_anchor_point(feat_sizes, self.fpn_strides, 0.5)
            yolox_losses = self.get_loss([
                cls_score_list, bbox_pred_list, obj_score_list, anchor_points,
                stride_tensor, num_anchors_list
            ], targets)
            return yolox_losses
        else:
            pred_scores = (cls_score_list * obj_score_list).sqrt()
            return pred_scores, bbox_pred_list, stride_tensor

    def get_loss(self, head_outs, targets):
        pred_cls, pred_bboxes, pred_obj,\
        anchor_points, stride_tensor, num_anchors_list = head_outs
        gt_labels = targets['gt_class']
        gt_bboxes = targets['gt_bbox']

        #gt_labels = paddle.cast(gt_labels, 'float32')
        #gt_bboxes = paddle.cast(gt_bboxes, 'float32')
        pred_scores = (pred_cls * pred_obj).sqrt()
        # label assignment
        center_and_strides = paddle.concat(
            [anchor_points, stride_tensor, stride_tensor], axis=-1)
        pos_num_list, label_list, bbox_target_list = [], [], []
        for pred_score, pred_bbox, gt_box, gt_label in zip(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor, gt_bboxes, gt_labels):
            ### TODO
            try:
                gt_box = paddle.cast(gt_box, 'float32')
            except:
                embed()
            gt_label = paddle.cast(gt_label, 'float32')
            ###
            pos_num, label, _, bbox_target = self.assigner(
                pred_score, center_and_strides, pred_bbox, gt_box, gt_label)
            pos_num_list.append(pos_num)
            label_list.append(label)
            bbox_target_list.append(bbox_target)
        labels = paddle.to_tensor(np.stack(label_list, axis=0))
        bbox_targets = paddle.to_tensor(np.stack(bbox_target_list, axis=0))
        bbox_targets /= stride_tensor  # rescale bbox

        # 1. obj score loss
        mask_positive = (labels != self.num_classes)
        loss_obj = F.binary_cross_entropy(
            pred_obj,
            mask_positive.astype(pred_obj.dtype).unsqueeze(-1),
            reduction='sum')

        num_pos = sum(pos_num_list)

        if num_pos > 0:
            num_pos = paddle.to_tensor(num_pos, dtype=self._dtype).clip(min=1)
            loss_obj /= num_pos

            # 2. iou loss
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                bbox_targets, bbox_mask).reshape([-1, 4])
            bbox_iou = bbox_overlaps(pred_bboxes_pos, assigned_bboxes_pos)
            bbox_iou = paddle.diag(bbox_iou)

            loss_iou = self.iou_loss(
                pred_bboxes_pos.split(
                    4, axis=-1),
                assigned_bboxes_pos.split(
                    4, axis=-1))
            loss_iou = loss_iou.sum() / num_pos

            # 3. cls loss
            cls_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.num_classes])
            pred_cls_pos = paddle.masked_select(
                pred_cls, cls_mask).reshape([-1, self.num_classes])
            assigned_cls_pos = paddle.masked_select(labels, mask_positive)
            assigned_cls_pos = F.one_hot(assigned_cls_pos,
                                         self.num_classes + 1)[..., :-1]
            assigned_cls_pos *= bbox_iou.unsqueeze(-1)
            loss_cls = F.binary_cross_entropy(
                pred_cls_pos, assigned_cls_pos, reduction='sum')
            loss_cls /= num_pos

            # 4. l1 loss
            if 1:  #targets['epoch_id'] >= self.l1_epoch:
                loss_l1 = F.l1_loss(
                    pred_bboxes_pos, assigned_bboxes_pos, reduction='sum')
                loss_l1 /= num_pos
            else:
                loss_l1 = paddle.zeros([1])
                loss_l1.stop_gradient = False
        else:
            loss_cls = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_l1 = paddle.zeros([1])
            loss_cls.stop_gradient = False
            loss_iou.stop_gradient = False
            loss_l1.stop_gradient = False

        loss = self.loss_weight['obj'] * loss_obj + \
               self.loss_weight['cls'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou * self.loss_weight['reg']

        if 1:  #targets['epoch_id'] >= self.l1_epoch:
            loss += (self.loss_weight['l1'] * loss_l1)

        yolox_losses = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_obj': loss_obj,
            'loss_iou': loss_iou,
            'loss_l1': loss_l1,
        }
        return yolox_losses

    def post_process(self, head_outs, img_shape, scale_factor):
        pred_scores, pred_bboxes, stride_tensor = head_outs
        pred_scores = pred_scores.transpose([0, 2, 1])
        pred_bboxes *= stride_tensor
        # scale bbox to origin image
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num

    ### TODO
    def forward2(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"
        cls_score_list, reg_pred_list = [], []
        obj_score_list = []
        xs = []
        zs = []
        #### [8, 64, 80, 80] [8, 128, 40, 40] [8, 256, 20, 20]
        for i, feat in enumerate(feats):
            feat = self.stem_conv[i](feat)
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            cls_output = self.cls_preds[i](cls_feat)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)
            #reg_output, obj_output = paddle.split(reg_pred, [4, 1], axis=1)

            if self.training:
                x = paddle.concat([reg_output, obj_output, cls_output], 1)
                bs, _, ny, nx = x.shape
                x = x.reshape([bs, self.na, self.no, ny, nx])
                x = paddle.transpose(x, [0, 1, 3, 4, 2])
                xs.append(x)
            else:
                y = paddle.concat(
                    [reg_output, F.sigmoid(obj_output), F.sigmoid(cls_output)],
                    1)
                bs, _, ny, nx = y.shape  # 1, _, 80, 80
                y = y.reshape([bs, self.na, self.no, ny, nx])
                y = paddle.transpose(y, [0, 1, 3, 4, 2])  # [1, 1, 80, 80, 85]

                if self.grid[i].shape[2:4] != y.shape[2:4]:
                    yv, xv = paddle.meshgrid(
                        [paddle.arange(ny), paddle.arange(nx)])
                    self.grid[i] = paddle.stack([xv, yv], 2).reshape(
                        [1, self.na, ny, nx, 2])
                    # [1, 1, 80, 80, 2]
                xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                wh = paddle.exp(y[..., 2:4]) * self.stride[i]  # wh
                y = paddle.concat((xy, wh, y[..., 4:]), -1)
                zs.append(y.reshape([bs, -1, self.no]))

        return xs if self.training else paddle.concat(zs, 1)

    def post_process2(self, head_outs, img_shape, scale_factor):
        bbox_list, score_list = [], []
        '''
            # confidence multiply the objectness
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
        '''

        for i, head_out in enumerate(head_outs):
            bs, _, ny, nx = head_out.shape
            head_out = head_out.reshape(
                [bs, self.num_anchor, self.num_out_ch, ny, nx]).transpose(
                    [0, 1, 3, 4, 2])
            # head_out.shape [bs, self.num_anchor, ny, nx, self.num_out_ch]

            bbox, score = self.postprocessing_by_level(head_out, self.stride[i],
                                                       self.anchors[i], ny, nx)
            bbox = bbox.reshape([bs, self.num_anchor * ny * nx, 4])
            score = score.reshape(
                [bs, self.num_anchor * ny * nx, self.num_classes]).transpose(
                    [0, 2, 1])
            bbox_list.append(bbox)
            score_list.append(score)

        pred_bboxes = paddle.concat(bbox_list, axis=1)
        pred_scores = paddle.concat(score_list, axis=-1)
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor

        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark for speed test
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])],
                axis=-1), paddle.to_tensor(
                    [1], dtype='int32')
        else:
            bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            return bbox_pred, bbox_num

    def post_process1(self, head_outs, img_shape, scale_factor):
        bbox_preds = non_max_suppression(
            head_outs,
            conf_thres=0.001,
            iou_thres=0.65,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=300)
        bbox_pred = bbox_preds[0]
        if len(bbox_pred.shape) == 1:
            bbox_pred = bbox_pred.unsqueeze(0)

        bbox_pred = paddle.concat(
            [bbox_pred[:, 5:6], bbox_pred[:, 4:5], bbox_pred[:, 0:4]], 1)
        bbox_num = paddle.to_tensor([len(bbox_pred)])
        return bbox_pred, bbox_num

    # def post_process(self, head_outs, img_shape, scale_factor):
    #     bbox_preds = non_max_suppression(head_outs, conf_thres=0.001, iou_thres=0.65, classes=None, agnostic=False, multi_label=False, max_det=300)
    #     bbox_pred = bbox_preds[0]
    #     if len(bbox_pred.shape) == 1:
    #         bbox_pred = bbox_pred.unsqueeze(0)

    #     bbox_pred = paddle.concat([bbox_pred[:, 5:6], bbox_pred[:, 4:5], bbox_pred[:, 0:4]], 1)
    #     bbox_num = paddle.to_tensor([len(bbox_preds)])
    #     return bbox_pred, bbox_num

    # pred_scores = pred_scores.transpose([0, 2, 1])
    # pred_bboxes *= stride_tensor
    # # scale bbox to origin image
    # scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
    # pred_bboxes /= scale_factor
    # if self.exclude_nms:
    #     # `exclude_nms=True` just use in benchmark
    #     return pred_bboxes.sum(), pred_scores.sum()
    # else:
    #     bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
    #     return bbox_pred, bbox_num


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """
    #outputs = []
    #for i in range(len(predictions)):
    if 1:
        num_classes = prediction.shape[2] - 5  # number of classes
        pred_candidates = prediction[..., 4] > conf_thres  # candidates

        # Check the parameters.
        assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
        assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
        multi_label &= num_classes > 1  # multiple labels per box

        output = [paddle.zeros([0, 6])] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(
                    as_tuple=False).T
                x = paddle.concat(
                    (box[box_idx], x[box_idx, class_idx + 5].unsqueeze(-1),
                     class_idx.unsqueeze(-1) * 1.0), 1)
            else:  # Only keep the class with highest scores.
                conf = x[:, 5:].max(1, keepdim=True)
                class_idx = x[:, 5:].argmax(1, keepdim=True)
                x = paddle.concat((box, conf, class_idx * 1.0),
                                  1)[conf[:, 0] > conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if classes is not None:
                x = x[(x[:, 5:6] == paddle.tensor(classes)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(
                    descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :
                              4] + class_offset, x[:,
                                                   4]  # boxes (offset by class), scores

            boxes = torch.tensor(boxes.numpy())
            scores = torch.tensor(scores.numpy())

            # [1366, 4] [1366, 1]
            keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            keep_box_idx = paddle.to_tensor(keep_box_idx.numpy())
            output[img_idx] = x[keep_box_idx]

    return output


def xywh2xyxy(x):
    # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
    y = x.clone() if isinstance(x, paddle.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
