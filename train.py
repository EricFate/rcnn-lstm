from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from tqdm import tqdm
import torch as t
from torch import nn

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from data.Data import COCODataset
from model import FasterRCNNVGG16, FasterRCNNVGG16Extractor
from faster_rcnn_lstm import FasterRCNN_LSTM
from torch.utils import data as data_
from trainer import FasterRCNNTrainer,LSTMTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from model.predictor import PredictNet
import pandas as pd
import numpy as np

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667


matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result




def pretrain(embedding_file):
    dataset = COCODataset(embedding_file, opt, True)
    opt.n_class = dataset.n_class
    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn_trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    opt.caffe_pretrain = False
    lstm = nn.LSTM(input_size=dataset.word_embedding.vector_size, hidden_size=opt.hidden_size, batch_first=True)
    predict_param = [lstm.hidden_size, opt.n_class]
    text_predictor = PredictNet(predict_param)
    text_predictor.parameters()
    lstm_trainer = LSTMTrainer(lstm, text_predictor).cuda()
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    for epoch in range(opt.epoch):
        for ii, (img, bbox_,text, label_, scale) in tqdm(enumerate(dataloader)):
            # train faster rcnn
            scale = at.scalar(scale)
            img, bbox,text, label = img.cuda().float(),bbox_.cuda(),text.cuda(), label_.cuda()
            faster_rcnn_trainer.train_step(img, bbox, label, scale)
            # train lstm
            lstm_trainer.lstm_step(text, label)
    state = {'rcnn':{'model':faster_rcnn_trainer.state_dict(),'optimizer':faster_rcnn_trainer.optimizer.state_dict()},
             'lstm': {'model': lstm_trainer.state_dict(), 'optimizer': lstm_trainer.optimizer.state_dict()}}

    t.save(state,'pretrain.pth')

        #     if (ii + 1) % opt.plot_every == 0:
        #         if os.path.exists(opt.debug_file):
        #             ipdb.set_trace()
        #
        #         # plot loss
        #         trainer.vis.plot_many(trainer.get_meter_data())
        #
        #         # plot groud truth bboxes
        #         ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        #         gt_img = visdom_bbox(ori_img_,
        #                              at.tonumpy(bbox_[0]),
        #                              at.tonumpy(label_[0]))
        #         trainer.vis.img('gt_img', gt_img)
        #
        #         # plot predicti bboxes
        #         _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        #         pred_img = visdom_bbox(ori_img_,
        #                                at.tonumpy(_bboxes[0]),
        #                                at.tonumpy(_labels[0]).reshape(-1),
        #                                at.tonumpy(_scores[0]))
        #         trainer.vis.img('pred_img', pred_img)
        #
        #         # rpn confusion matrix(meter)
        #         trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
        #         # roi confusion matrix
        #         trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        # eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # trainer.vis.plot('test_map', eval_result['map'])
        # lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        # log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
        #                                           str(eval_result['map']),
        #                                           str(trainer.get_meter_data()))
        # trainer.vis.log(log_info)
        #
        # if eval_result['map'] > best_map:
        #     best_map = eval_result['map']
        #     best_path = trainer.save(best_map=best_map)
        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
        #     lr_ = lr_ * opt.lr_decay
        #
        # if epoch == 13:
        #     break

def _train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)




            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            break


def evaluate(dataset, faster_rcnn_lstm):
    val_imgs, val_texts, val_bboxes, val_labels = dataset.get_test_data()
    average_precison, coverage, example_auc, macro_auc, micro_auc, ranking_loss = faster_rcnn_lstm.test(val_imgs, val_texts, val_labels,opt.m,opt.k)
    return average_precison, coverage, example_auc, macro_auc, micro_auc, ranking_loss

def train():
    embedding_file = 'G:\\data\\coco_filtered_word2vec_1024'
    dataset = COCODataset(embedding_file,False,opt)
    opt.n_class = dataset.n_class
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # trainer.load('G:\\models\\fasterrcnn_torchvision_pretrain.pth')
    opt.caffe_pretrain = False
    faster_extractor = FasterRCNNVGG16Extractor(trainer.faster_rcnn, opt.n_class, hidden_size=opt.hidden_size).cuda()
    lstm = nn.LSTM(input_size=dataset.word_embedding.vector_size, hidden_size=opt.hidden_size, batch_first=True)
    predict_param = [dataset.word_embedding.vector_size * 4, opt.n_class]
    final_predictor = PredictNet(predict_param)
    faster_rcnn_lstm = FasterRCNN_LSTM(faster_extractor, lstm, final_predictor)
    test_result = list()
    for epoch in range(opt.epoch):
        for ii, (image,text,bbox,labels) in tqdm(enumerate(dataloader)):
            faster_rcnn_lstm.train_step(image, text, bbox, labels,opt.m,opt.k)
        result = evaluate(dataset,faster_rcnn_lstm)
        test_result.append(result)
    test_result = pd.DataFrame(data=np.array(test_result),columns=['average_precison', 'coverage', 'example_auc', 'macro_auc', 'micro_auc', 'ranking_loss'])
    test_result.to_csv('test_result')



if __name__ == '__main__':
    embedding_file = 'coco_filtered_word2vec_1024'
    pretrain(embedding_file)
