# -*- coding: utf-8 -*-
"""
Created on 2023/8/7 2:00 
@Author: Wu Kaixuan
@File  : utils.py 
@Desc  : utils 
"""
import os
import json
from collections import Counter
import numpy as np
import mindspore as ms
import pandas as pd

import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import sys
import logging


class CustomLogger(object):
    __FORMATTER__ = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    def __init__(self, log_file, formatter=__FORMATTER__, log_level=logging.DEBUG):
        self.log_file = log_file
        self.formatter = formatter
        self.log_level = log_level

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def get_file_handler(self):
        file_handle = logging.FileHandler(self.log_file, mode='a+', encoding='utf-8')
        file_handle.setFormatter(self.formatter)
        return file_handle

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        logger.propagate = False
        return logger



def extrct_split(dataset,
                 json_path,
                 min_word_freq=5,
                 captions_per_image=5,
                 output_folder="output",
                 max_len=100):
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Read Karpathy JSON
    with open(json_path, 'r') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
        if len(captions) == 0:
            continue
        path = img['filename']
        if img['split'] in {'train', 'restval'}:
            for i in range(len(captions)):
                train_image_paths.append(path)
                train_image_captions.append(captions[i])
        elif img['split'] in {'val'}:
            for i in range(len(captions)):
                val_image_paths.append(path)
                val_image_captions.append(captions[i])
        elif img['split'] in {'test'}:
            for i in range(len(captions)):
                test_image_paths.append(path)
                test_image_captions.append(captions[i])

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    pd.DataFrame({'image_path': train_image_paths, 'captions': train_image_captions}).to_csv(
        os.path.join(output_folder, 'train.csv'), index=False)
    pd.DataFrame({'image_path': val_image_paths, 'captions': val_image_captions}).to_csv(
        os.path.join(output_folder, 'val.csv'), index=False)
    pd.DataFrame({'image_path': test_image_paths, 'captions': test_image_captions}).to_csv(
        os.path.join(output_folder, 'test.csv'), index=False)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<sos>'] = len(word_map) + 1
    word_map['<eos>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, valid_lengths, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param preds: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.shape[0]
    # _, ind = preds.topk(k, 2, True, True)
    correct_counts = 0
    total_counts = 0
    for i, length in enumerate(valid_lengths):
        valid_pred = preds[i, :length]
        valid_target = targets[i, :length]
        topk_vals, topk_idx = valid_pred.topk(k, 1)
        # 检查目标是否在前5的索引中
        correct = topk_idx.equal(valid_target.unsqueeze(-1).expand_as(topk_idx)).sum()
        correct_counts += correct.float().sum()
        total_counts += length
    # 计算top-5准确率
    top5_accuracy = correct_counts / total_counts
    return top5_accuracy


if __name__ == '__main__':
    extrct_split(dataset='flickr8k',
                 json_path="flickr8k/dataset.json",
                 # image_folder=r"F:\ShowandTell\Flicker8k_Dataset",
                 captions_per_image=5,
                 min_word_freq=5,
                 output_folder="flickr8k",
                 max_len=50
                 )

