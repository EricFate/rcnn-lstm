import torch as t
from torch import nn
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from utils import array_tool as at
from utils.config import opt
from model.faster_rcnn import nograd
from test import measure_average_precision, measure_coverage, measure_example_auc, measure_macro_auc, measure_micro_auc, \
    measure_ranking_loss


class FasterRCNN_LSTM(nn.Module):
    def __init__(self, faster_rcnn, lstm, final_predictor):
        super(FasterRCNN_LSTM, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.lstm = lstm
        self.n_class = faster_rcnn.n_class
        self.final_predictor = final_predictor

        self.bce = nn.BCELoss()
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()



    def _rcnn_step(self, imgs, bboxes, labels, m):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """

        n = len(imgs)
        losses_total = list()
        features, roi_cls_locs, roi_scores, rpn_locs, rpn_scores, rois, anchors, scales, hiddens = self.faster_rcnn.extract(
            imgs, m)
        if labels is None:
            return features
        for i in range(n):
            # Since batch size is one, convert variables to singular form
            img = imgs[i]
            _, H, W = img.shape
            img_size = (H, W)
            bbox = bboxes[i]
            label = labels[i]
            rpn_score = rpn_scores[i]
            rpn_loc = rpn_locs[i]
            roi = rois[i]
            h = hiddens[i]
            anchor = anchors[i]
            # Sample RoIs and forward
            # it's fine to break the computation graph of rois,
            # consider them as constant input
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi,
                at.tonumpy(bbox),
                at.tonumpy(label),
                self.loc_normalize_mean,
                self.loc_normalize_std)
            # NOTE it's all zero because now it only support for batch=1 now
            sample_roi_index = t.zeros(len(sample_roi))
            roi_cls_loc, roi_score, _ = self.faster_rcnn.head(
                h,
                sample_roi,
                sample_roi_index)

            # ------------------ RPN losses -------------------#
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(bbox),
                anchor,
                img_size)
            gt_rpn_label = at.totensor(gt_rpn_label).long()
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma)

            # NOTE: default value of ignore_index is -100 ...
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
            _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
            _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
            self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

            # ------------------ ROI losses (fast rcnn loss) -------------------#
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                                  at.totensor(gt_roi_label).long()]
            gt_roi_label = at.totensor(gt_roi_label).long()
            gt_roi_loc = at.totensor(gt_roi_loc)

            roi_loc_loss = _fast_rcnn_loc_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                self.roi_sigma)

            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

            self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
            losses = sum(losses)
            losses_total.append(losses)
        return t.mean(losses_total), features

    def _extract_common_unique_feature(self, regions, sigments, k):
        region_cons = list()
        sigment_cons = list()
        region_uniques = list()
        sigment_uniques = list()
        for (r, s) in zip(regions, sigments):
            sim_mat = t.matmul(r, s.t())
            sim_mat_flat = t.flatten(sim_mat)
            kth, _ = t.kthvalue(sim_mat_flat, len(sim_mat_flat) - 30)
            idx_k = sim_mat > kth
            m_idx = (t.sum(idx_k, 1) > 0).float()
            n_idx = (t.sum(idx_k, 0) > 0).float()
            cons_select_m = m_idx.view(-1, 1).repeat(1, self.hidden_size)
            cons_select_n = n_idx.view(-1, 1).repeat(1, self.hidden_size)
            cons_image = r * cons_select_m
            cons_text = s * cons_select_n
            cons_image = t.sum(cons_image, 0)
            cons_text = t.sum(cons_text, 0)
            unique_image = r * (1 - cons_select_m)
            unique_text = s * (1 - cons_select_n)
            region_cons.append(cons_image)
            sigment_cons.append(cons_text)
            region_uniques.append(unique_image)
            sigment_uniques.append(unique_text)
        region_cons = t.cat(region_cons)
        sigment_cons = t.cat(sigment_cons)
        region_uniques = t.cat(region_uniques)
        sigment_uniques = t.cat(sigment_uniques)

        return region_cons, sigment_cons, region_uniques, sigment_uniques

    def train_step(self, imgs, texts, bboxes, labels, m, k):
        self.optimizer.zero_grad()
        rcnn_loss, region_supervise_feature = self._rcnn_step(imgs[0], bboxes, labels, m)
        lstm_loss, sigment_supervise_feature = self._lstm_step(texts[0], labels)
        region_unsupervise_feature = self._rcnn_step(imgs[1], None, None, m)
        sigment_unsupervise_feature = self._lstm_step(texts[1], None)
        region_supervise_cons, sigment_supervise_cons, region_supervise_uniques, sigment_supervise_uniques = self._extract_common_feature(
            region_supervise_feature, sigment_supervise_feature, k)
        assemble_feature = t.cat(
            (region_supervise_cons, sigment_supervise_cons, region_supervise_uniques, sigment_supervise_uniques))
        assemble_predict = self.final_predictor(assemble_feature)
        feature_loss = self.bce(assemble_predict, labels)
        region_unsupervise_cons, sigment_unsupervise_cons, region_unsupervise_uniques, sigment_unsupervise_uniques = self._extract_common_feature(
            region_unsupervise_feature, sigment_unsupervise_feature, k)
        region_cons = t.cat((region_supervise_cons, region_unsupervise_cons))
        sigment_cons = t.cat((sigment_supervise_cons, sigment_unsupervise_cons))
        consistency_loss = 1 - nn.CosineSimilarity()(region_cons, sigment_cons)
        loss = consistency_loss + feature_loss
        loss.backward()
        self.optimizer.step()
        return loss

    def _to_one_hot(self, labels):
        shape = (len(labels), self.n_class)
        one_hot = t.zeros(shape).long()
        for i in range(len(labels)):
            one_hot[i, labels[i]] = 1
        return one_hot

    @nograd
    def test(self, imgs, texts, labels, m, k):
        region_supervise_feature = self._rcnn_step(imgs, None, None, m)
        sigment_supervise_feature = self._lstm_step(texts, None)
        region_supervise_cons, sigment_supervise_cons, region_supervise_uniques, sigment_supervise_uniques = self._extract_common_feature(
            region_supervise_feature, sigment_supervise_feature, k)
        assemble_feature = t.cat(
            (region_supervise_cons, sigment_supervise_cons, region_supervise_uniques, sigment_supervise_uniques))
        assemble_predict = self.final_predictor(assemble_feature)
        one_hot_label = self._to_one_hot(labels)

        average_precison = measure_average_precision.average_precision(assemble_predict, one_hot_label)
        coverage = measure_coverage.coverage(assemble_predict, one_hot_label)
        example_auc = measure_example_auc.example_auc(assemble_predict, one_hot_label)
        macro_auc = measure_macro_auc.macro_auc(assemble_predict, one_hot_label)
        micro_auc = measure_micro_auc.micro_auc(assemble_predict, one_hot_label)
        ranking_loss = measure_ranking_loss.ranking_loss(assemble_predict, one_hot_label)

        return average_precison,coverage,example_auc,macro_auc,micro_auc,ranking_loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss
