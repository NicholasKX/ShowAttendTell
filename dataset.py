# -*- coding: utf-8 -*-
"""
Created on 2023/8/7 1:37 
@Author: Wu Kaixuan
@File  : dataset.py 
@Desc  : dataset 
"""
import re
import os
from typing import *
import pandas as pd
import numpy as np
from PIL import Image
from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds
from mindspore.dataset import text
import json
import mindspore.dataset.transforms as transforms


# def build_vocabulary(json_file):
#     special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
#     word2idx = json.load(open(json_file, 'r'))
#
#     # json key to list
#     vocab_list = [k for k, v in word2idx.items() if k not in special_tokens]
#     vocab_len = len(vocab_list)+4
#     vocab = text.Vocab.from_list(vocab_list, special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])
#     setattr(vocab, "vocab_len", vocab_len)
#     return vocab

def build_vocabulary(json_file):
    special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    word2idx = json.load(open(json_file, 'r'))

    vocab = text.Vocab.from_dict(word2idx)

    setattr(vocab, "vocab_len", len(word2idx))
    return vocab


class Vocabulary:
    def __init__(self, vocab_file):
        """

        :param vocab_file: json file
        """
        self.word2idx = json.load(open(vocab_file, 'r'))
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_len = len(self.word2idx)

    def tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            if tokens not in self.word2idx:
                return self.word2idx["<unk>"]
            else:
                return self.word2idx[tokens]
        elif isinstance(tokens, list):
            res = []
            for token in tokens:
                if token not in self.word2idx:
                    res.append(self.word2idx["<unk>"])
                else:
                    res.append(self.word2idx[token])
            return res

    def ids_to_tokens(self, ids):
        if isinstance(ids, int):
            if ids not in self.idx2word:
                return "<unk>"
            else:
                return self.idx2word[ids]
        elif isinstance(ids, list):
            res = []
            for id in ids:
                if id not in self.idx2word:
                    res.append("<unk>")
                else:
                    res.append(self.idx2word[id])
            return res

    def __len__(self):
        return self.vocab_len


class FlickrDataset:
    def __init__(self,
                 data_dir,
                 img_dir,
                 split="train",
                 vocab=None,
                 transform=None):
        if split == "train":
            self.data = pd.read_csv(os.path.join(data_dir, "train.csv"))
            print(f"Load train data from {os.path.join(data_dir, 'train.csv')}")
        elif split == "test":
            self.data = pd.read_csv(os.path.join(data_dir, "test.csv"))
            print(f"Load test data from {os.path.join(data_dir, 'test.csv')}")
        elif split == "val":
            self.data = pd.read_csv(os.path.join(data_dir, "val.csv"))
            print(f"Load val data from {os.path.join(data_dir, 'val.csv')}")
        else:
            raise ValueError("split must be one of train, test, val")

        # self.img_root = os.path.join(data_dir, "Images")
        self.img_root = img_dir
        self.transform = transform
        self.vocab = vocab
        self.captions = self.data.captions.map(lambda x: eval(x))
        self.avg_len = np.mean([len(caption) for caption in self.captions])
        self.imgs = self.data.image_path
        self.split = split

    def __getitem__(self, index):
        img_path = os.path.join(self.img_root, self.imgs[index])
        caption = self.captions[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        tokenized_caption = [self.vocab.tokens_to_ids("<sos>")]
        tokenized_caption += self.vocab.tokens_to_ids(caption)
        tokenized_caption.append(self.vocab.tokens_to_ids("<eos>"))
        if self.split == "train":
            return img, np.array(tokenized_caption), len(tokenized_caption)
        elif self.split == "val" or self.split == "test":
            all_captions = self.captions[(index // 5) * 5:(index // 5) * 5 + 5]
            pad_op = transforms.PadEnd([40], pad_value=0)
            all_captions = [self.vocab.tokens_to_ids(caption) for caption in all_captions]
            all_captions = [pad_op(caption) for caption in all_captions]
            return img, np.array(tokenized_caption), len(tokenized_caption), all_captions

    def __len__(self):
        return self.data.shape[0]

    def get_avg_len(self):
        return self.avg_len


if __name__ == '__main__':
    # dataset = create_dataset()
    # print(dataset.get_dataset_size())
    # dataset = dataset.create_dict_iterator()
    # for data in dataset:
    #     print(data["image"])
    #     print(data["image"].shape)
    #     print(data["annotation"])
    #     print(data["annotation"].shape)
    #     break
    # vocab = build_vocabulary("flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json")
    vocab = Vocabulary("flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json")
    print(vocab.tokens_to_ids("<sos>"))
    print(vocab.tokens_to_ids("<eos>"))
    print(vocab.tokens_to_ids("<pad>"))
    print(vocab.tokens_to_ids("<unk>"))

    # print(vocab.vocab())
    # dataset = create_dataset()
    # dataset = dataset.map(text.Lookup(vocab))
    print(vocab.tokens_to_ids("lalaldax"))
    dataset = FlickrDataset(data_dir=r"flickr8k",
                            img_dir=r"F:\ShowandTell\Flicker8k_Dataset",
                            split="val", vocab=vocab)
    print(dataset.get_avg_len())
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[0][1])
    print(dataset[0][2])
    print(dataset[0][3])
