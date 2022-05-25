import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
import numpy as np


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


@register
class YOLOv3Head(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 data_format='NCHW'):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format

        self.yolo_outputs = []
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)
        self._initialize_biases()

    def _initialize_biases(self):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        import math
        stride = [8,16,32]
        num_a = 3
        for i,conv in enumerate(self.yolo_outputs):
            b = conv.bias.numpy().reshape([3,-1])
            b[:, 4] += math.log(8 / (640 / stride[i]) ** 2)
            b[:, 5:] += math.log(0.6 / (num_a - 0.999999))
            conv.bias.set_value(b.reshape([-1]))
            # pass

        # m = self.model[-1]  # Detect() module
        # for mi, s in zip(m.m, m.stride):  # from
        #     b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        #     b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #     b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
        #     mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        #print('yolo_outputs ', [x.shape for x in yolo_outputs])
        #print('yolo_outputs sum ', [x.sum() for x in yolo_outputs])

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj**(1 - self.iou_aware_factor)) * (
                        ioup**self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    
    
@register
class YOLOv5Head(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss','nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 data_format='NCHW',
                 nms='nms',
                 stride=[8,16,32]):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv5Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.anchors = paddle.to_tensor(self.anchors,dtype='float32')
        self.num_outputs = len(self.anchors)
        self.data_format = data_format
        self.nms = nms
        self.stride = stride
        self.na = len(stride)
        self.no = self.num_classes + 5

        self.yolo_outputs = []
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)
#         self._initialize_biases()

    def _initialize_biases(self):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        import math
        stride = [8,16,32]
        for i,conv in enumerate(self.yolo_outputs):
            b = conv.bias.numpy().reshape([3,-1])
            # print(b.sum())
            b[:, 4] += math.log(8 / (640 / stride[i]) ** 2)
            b[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999))

            conv.bias.set_value(b.reshape([-1]))
            # print(math.log(0.6 / (num_c - 0.999999)))
#             print(conv.bias.sum())
            # print(conv.bias.stop_gradient)
            # pass
            # pass

        # m = self.model[-1]  # Detect() module
        # for mi, s in zip(m.m, m.stride):  # from
        #     b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        #     b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #     b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
        #     mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        #print('yolo_outputs ', [x.shape for x in yolo_outputs])
        #print('yolo_outputs sum ', [x.sum() for x in yolo_outputs])

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj**(1 - self.iou_aware_factor)) * (
                        ioup**self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def post_process(self, head_outs, img_shape, scale_factor):
        yolo_head_outs = head_outs
        bbox_list, score_list = [],[]
        for i, head_out in enumerate(yolo_head_outs):
            bs, _, ny, nx = head_out.shape
            head_out = head_out.reshape([bs, self.na, self.no, ny, nx]).transpose([0, 1, 3, 4, 2])
            bbox,score = self.postprocessing_by_level(head_out,self.stride[i],self.anchors[i],ny, nx)
            bbox = bbox.reshape([bs,-1,4])
            score = score.reshape([bs,-1,self.num_classes]).transpose([0,2,1])
            bbox_list.append(bbox)
            score_list.append(score)
        pred_bboxes = paddle.concat(bbox_list,axis=1)
        pred_scores = paddle.concat(score_list,axis=-1)
        scale_factor = scale_factor.flip(-1).tile([1, 2]).unsqueeze(1)
        pred_bboxes /= scale_factor

        bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
        bbox_coords = bbox_pred
        # bbox_coords = self.scale_coords(bbox_pred, img_shape, scale_factor)
        # num_id, score, xmin, ymin, xmax, ymax
        return bbox_coords, bbox_num

    def scale_coords(self, coords, im_shape, scale_factor):
        clses, scores, bboxes = coords.split([1,1,-1],axis=-1)
        img0_shape = im_shape[0]
        scale_h_ratio = scale_factor[:, 0]
        scale_w_ratio = scale_factor[:, 1]
        ratio = min(scale_h_ratio, scale_w_ratio)

        pad_w = (img0_shape[1] * scale_w_ratio - img0_shape[1] * ratio) / 2
        pad_h = (img0_shape[0] * scale_h_ratio - img0_shape[0] * ratio) / 2
        bboxes[..., 0::2] -= pad_w
        bboxes[..., 1::2] -= pad_h
        bboxes /= ratio

        bboxes[:, 0::2] = paddle.clip(bboxes[:, 0::2], min=0, max=img0_shape[1])
        bboxes[:, 1::2] = paddle.clip(bboxes[:, 1::2], min=0, max=img0_shape[0])
        return paddle.concat((clses, scores, bboxes), axis=-1)
    
    def postprocessing_by_level(self, head_out, stride, anchor, ny, nx):
        grid, anchor_grid = self.make_grid(nx, ny, anchor)
        out = F.sigmoid(head_out)
        xy = (out[..., 0:2] * 2. - 0.5 + grid) * stride
        wh = (out[..., 2:4] * 2) ** 2 * anchor_grid
        lt_xy = (xy - wh / 2.)
        rb_xy = (xy + wh / 2.)
        bboxes = paddle.concat((lt_xy, rb_xy), axis=-1)
        scores = out[...,5:]*out[...,4].unsqueeze(-1)

        return bboxes,scores

    def make_grid(self, nx, ny, anchor):
        yv, xv = paddle.meshgrid([paddle.arange(ny), paddle.arange(nx)])

        grid = paddle.stack((xv, yv), axis=2).expand([1, self.na, ny, nx, 2])
#         anchor_grid =  paddle.to_tensor(np.array(anchor).astype(np.float32)).reshape([1,self.na, 1, 1, 2]).expand((1, self.na, ny, nx, 2))
        anchor_grid = anchor.reshape([1, self.na, 1, 1, 2]).expand((1, self.na, ny, nx, 2))

        return grid, anchor_grid
