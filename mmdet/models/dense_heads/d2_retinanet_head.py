import torch.nn as nn
import torch
import math
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple
from .anchor_head import AnchorHead
from ..builder import HEADS, build_loss
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform


@HEADS.register_module()
class D2RetinaNetHeadx(AnchorHead):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super(D2RetinaNetHeadx, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        del self.conv_cls
        del self.conv_reg

        size = [[32, 40.31747359663594, 50.79683366298238],
                [64, 80.63494719327188, 101.59366732596476],
                [128, 161.26989438654377, 203.18733465192952],
                [256, 322.53978877308754, 406.37466930385904],
                [512, 645.0795775461751, 812.7493386077181]]
        self.anchor_generator = DefaultAnchorGenerator(sizes=size, aspect_ratios=[[0.5, 1.0, 2.0]],
                                                       strides=[8, 16, 32, 64, 128], offset=0)

        prior_prob = 0.01
        norm = ""
        conv_dims = [256, 256, 256, 256]
        num_classes = 80

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip([256] + conv_dims, conv_dims):
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], 9 * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], 9 * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@HEADS.register_module()
class D2RetinaNetHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super(D2RetinaNetHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        del self.conv_cls
        del self.conv_reg

        # Note：mmdet 和 d2 虽然anchor 生成的公式是一样的，但是内部排列顺序不一样，导致如果直接迁移 d2 权重，也必须用 d2 的 anchor_gen
        size = [[32, 40.31747359663594, 50.79683366298238],
                [64, 80.63494719327188, 101.59366732596476],
                [128, 161.26989438654377, 203.18733465192952],
                [256, 322.53978877308754, 406.37466930385904],
                [512, 645.0795775461751, 812.7493386077181]]
        self.anchor_generator = DefaultAnchorGenerator(sizes=size, aspect_ratios=[[0.5, 1.0, 2.0]],
                                                       strides=[8, 16, 32, 64, 128], offset=0)

        self.box2box_transform = Box2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0])

        prior_prob = 0.01
        norm = ""
        conv_dims = [256, 256, 256, 256]
        num_classes = 80

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip([256] + conv_dims, conv_dims):
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], 9 * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], 9 * 4, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg

    def get_bboxes(self,
                   pred_logits,
                   pred_anchor_deltas,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(pred_logits[0].shape[0])
        ]
        anchors = self.anchor_generator(pred_logits)

        # featmap_sizes = [pred_logits[i].shape[-2:] for i in range(5)]
        # anchors = self.anchor_generator.grid_anchors(
        #     featmap_sizes, device=pred_logits[0].device)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        results = self.inference(anchors, pred_logits, pred_anchor_deltas, scale_factors)
        return results

    def inference(
            self,
            anchors,
            pred_logits,
            pred_anchor_deltas,
            image_sizes,
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, scale_factor in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, scale_factor
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
            self,
            anchors,
            box_cls,
            box_delta,
            scale_factor,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > 0.05
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(1000, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            if hasattr(anchors_i, 'tensor'):
                predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
            else:
                predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, 0.5)
        keep = keep[: 100]

        boxes_all /= boxes_all.new_tensor(scale_factor)

        class_idxs_all = class_idxs_all[keep]
        scores_all = scores_all[keep]
        boxes_all = boxes_all[keep]
        dets = torch.cat([boxes_all, scores_all[:, None]], dim=1)

        return dets, class_idxs_all
