import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


@BBOX_ASSIGNERS.register_module()
class UniformAssigner(BaseAssigner):

    def __init__(self,
                 pos_ignore_thresh,
                 neg_ignore_thresh,
                 match_times=4,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bbox_pred, anchor, gt_bboxes, gt_labels, img_meta):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(bbox_pred), box_xyxy_to_cxcywh(gt_bboxes), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchor), box_xyxy_to_cxcywh(gt_bboxes), p=1)

        # TODO: TOPK function has different results in cpu and cuda mode
        C = cost_bbox.cpu()
        C1 = cost_bbox_anchors.cpu()

        # self.match_times x n
        index = torch.topk(
            C,  # c=b,n,x c[i]=n,x
            k=self.match_times,
            dim=0,
            largest=False)[1]

        # self.match_times x n
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        # (self.match_times*2) x n
        indexes = torch.cat((index, index1),
                            dim=1).reshape(-1).to(bbox_pred.device)

        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(anchor, gt_bboxes)
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)

        ignore_idx = pred_max_overlaps > self.neg_ignore_thresh
        assigned_gt_inds[ignore_idx] = -1

        pos_gt_index = torch.arange(
            0, C1.size(1),
            device=bbox_pred.device).repeat(self.match_times * 2)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thresh

        target_classes_o = pos_gt_index + 1
        target_classes_o[pos_ignore_idx] = -1
        target_classes_o = target_classes_o.to(assigned_gt_inds.device)

        # TODO
        # unique_src_idx = torch.unique(indexes)
        # target_classes = torch.zeros_like(
        #     unique_src_idx, device=indexes.device)
        # for i, idx in enumerate(unique_src_idx):
        #     index = indexes == idx
        #     max_index = torch.argmax(pos_ious[index])
        #     target_classes[i] = target_classes_o[index][max_index]
        # assigned_gt_inds[unique_src_idx] = target_classes.to(
        #     assigned_gt_inds.device)

        # GPU
        # assigned_gt_inds[indexes] = target_classes_o

        # CPU
        assigned_gt_inds = assigned_gt_inds.cpu()
        target_classes_o = target_classes_o.cpu()
        indexes = indexes.cpu()
        assigned_gt_inds[indexes] = target_classes_o
        assigned_gt_inds = assigned_gt_inds.to(pos_gt_index.device)

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result=AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,
            labels=assigned_labels)
        assign_result.set_extra_property('pos_idx',~pos_ignore_idx)
        assign_result.set_extra_property('pos_predicted_boxes',bbox_pred[indexes])
        assign_result.set_extra_property('target_boxes',gt_bboxes[pos_gt_index])
        return assign_result
