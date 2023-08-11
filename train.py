# -*- coding: utf-8 -*-
"""
Created on 2023/8/7 16:03 
@Author: Wu Kaixuan
@File  : train.py 
@Desc  : train 
"""
import time
from datetime import datetime
from random import seed
from typing import Literal
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
import mindspore.nn as nn
import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype, ms_function
from mindspore.nn.optim import optimizer
from tqdm import tqdm

from networks.models import Encoder, DecoderWithAttention, NetWithLossCell, TrainingWrapper
from dataset import FlickrDataset, build_vocabulary,Vocabulary
from mindspore.dataset import GeneratorDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu


LOG_DIR = "logs/"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
now = datetime.now()  # 获得当前时间
timestr = now.strftime("%Y%m%d%H%M%S_")
LOG_FILE = os.path.join(LOG_DIR, timestr + "train.log")
logger = CustomLogger(LOG_FILE).get_logger('train')


data_dir = "flickr8k/Images"
model_save_path = "model_saved"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
workers = 8  # 并行线程个数

# Training parameters
batch_size = 32  # 批量大小
start_epoch = 0
epochs = 200  # 训练轮数
epochs_since_improvement = 0  # 距离上次提升轮数

encoder_lr = 1e-4  # encoder的学习率
decoder_lr = 4e-4  # decoder的学习率
grad_clip = 5.  # 梯度裁剪
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # 目前BLEU-4分数
print_freq = 10  # 每训练多少个batch打印一次训练状态
fine_tune_encoder = False  # 是否微调encoder
encoder_checkpoint = "checkpoint/ms_encoder.ckpt"  # 模型路径
decoder_checkpoint = "model_saved/decoder_epoch_0_bleu4_0.34044173648134046.ckpt"  # 模型路径
ms.set_context(mode=ms.PYNATIVE_MODE)


# ms.set_context(device_target="CPU", mode=ms.GRAPH_MODE, save_graphs=False)
def create_dataset_flickr(data,
                          vocab,
                          split: Literal["train", "val","test"] = "train",
                          resize: int = 224,
                          batch_size: int = 4,
                          workers: int = 4):
    if split == "train":
        dataset = GeneratorDataset(data, ["image", "caption", "cap_len"])
    elif split == "val":
        dataset = GeneratorDataset(data, ["image", "caption", "cap_len", "all_captions"])
    elif split == "test":
        dataset = GeneratorDataset(data, ["image", "caption", "cap_len", "all_captions"])
    else:
        raise ValueError("split must be 'train' or 'val' or 'test'")
    trans = []
    trans += [
        vision.Resize((resize, resize)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ]
    pad_op = transforms.PadEnd([40], pad_value=vocab.tokens_to_ids("<pad>"))
    target_trans = transforms.TypeCast(mstype.int32)
    # 数据映射操作
    dataset = dataset.map(operations=trans,
                          input_columns='image',
                          num_parallel_workers=workers)

    dataset = dataset.map(operations=[pad_op, target_trans],
                          input_columns='caption',
                          num_parallel_workers=workers)
    dataset = dataset.map(operations=[target_trans],
                          input_columns='cap_len',
                          num_parallel_workers=workers)
    # 批量操作
    dataset = dataset.batch(batch_size)
    return dataset



# model = NetWithLossCell(encoder, decoder, loss_fn=criterion)
# optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr)
# config = CheckpointConfig(save_checkpoint_steps=step_size_train)
#
# ckpt_callback = ModelCheckpoint(prefix="flickr8k", directory="./checkpoint", config=config)
# loss_callback = LossMonitor(step_size_train)
# trainer = Model(model, optimizer=optimizer)
# # model = TrainingWrapper(model, optimizer)
# trainer.train(10, data, callbacks=[ckpt_callback, loss_callback])
# vocab = build_vocabulary("flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json")
vocab = Vocabulary("flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json")
vocab_size = len(vocab)
dataset = FlickrDataset(data_dir=r"flickr8k",
                        img_dir=data_dir,
                        split="train", vocab=vocab)
val_dataset = FlickrDataset(data_dir=r"flickr8k",
                            img_dir=data_dir,
                            split="val", vocab=vocab)
data = create_dataset_flickr(dataset, vocab, batch_size=batch_size)
val_data = create_dataset_flickr(val_dataset, vocab, split="val", batch_size=batch_size)
step_size_train = data.get_dataset_size()
step_size_val = val_data.get_dataset_size()
logger.info(f"train_data:{step_size_train}")
logger.info(f"val_data:{step_size_val}")

data_loader = data.create_tuple_iterator()
val_data_loader = val_data.create_tuple_iterator()

decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=vocab_size,
                               dropout=dropout)

encoder = Encoder()
criterion = nn.CrossEntropyLoss(ignore_index=vocab.tokens_to_ids("<pad>"))

lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=100 * 20,
                        step_per_epoch=100, decay_epoch=20)
decoder_optimizer = nn.Adam(params=decoder.trainable_params(), learning_rate=lr)
if encoder_checkpoint is not None:
    encoder_param_dict = ms.load_checkpoint(encoder_checkpoint)
    param_not_load, _ = ms.load_param_into_net(encoder, encoder_param_dict)
    logger.info("load encoder checkpoint success")
if decoder_checkpoint is not None:
    decoder_param_dict = ms.load_checkpoint(decoder_checkpoint)
    param_not_load, _ = ms.load_param_into_net(decoder, decoder_param_dict)
    logger.info("load decoder checkpoint success")


def main(train_epochs):
    max_bleu4 = 0
    for epoch in tqdm(range(train_epochs)):
        train_loop(data_loader, encoder, decoder, epoch)
        bleu4 = validate(val_data_loader, encoder, decoder)
        if bleu4 > max_bleu4:
            max_bleu4 = bleu4
            logger.info('saving model...')
            ms.save_checkpoint(encoder, f"{model_save_path}/encoder_epoch_{epoch}_bleu4_{max_bleu4:.4f}.ckpt")
            ms.save_checkpoint(decoder, f"{model_save_path}/decoder_epoch_{epoch}_bleu4_{max_bleu4:.4f}.ckpt")


def forward_fn(imgs, caps, caplens):

    imgs = encoder(imgs)
    scores, alphas, encoded_captions, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    # Remove timesteps that we didn't decode at, or are pads
    targets = caps[:, 1:max(caplens)]
    # Calculate loss
    targets = targets[sort_ind]
    loss = criterion(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1))
    # Add doubly stochastic attention regularization
    loss += alpha_c * ((1. - alphas.sum(axis=1)) ** 2).mean()
    return loss, scores, encoded_captions, decode_lengths, sort_ind


grad_fn = ms.value_and_grad(forward_fn, grad_position=None, weights=decoder_optimizer.parameters, has_aux=True)

def train_step(imgs, caps, caplens):
    (loss, scores, encoded_captions, decode_lengths, sort_ind), grads = grad_fn(imgs, caps, caplens)
    if grad_clip is not None:
        grads = ms.ops.clip_by_value(grads, -grad_clip, grad_clip)
    decoder_optimizer(grads)
    return loss, scores,encoded_captions, decode_lengths, sort_ind

def train_loop(train_loader,
               encoder,
               decoder,
               epoch,
               ):
    """
    Performs one epoch's training.
    :param grad_clip:  梯度裁剪
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    decoder.set_train()  # train mode (dropout and batchnorm is used)
    encoder.set_train(False) # encoder is always in eval mode

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()
    # Batches
    for i, (imgs, caps, caplens) in tqdm(enumerate(train_loader), total=step_size_train):
        data_time.update(time.time() - start)
        loss, scores, caps_sorted, decode_lengths, sort_ind = train_step(imgs, caps, caplens)
        # Keep track of metrics
        top5 = accuracy(scores, caps_sorted[:, 1:max(caplens)], caplens, 5)
        losses.update(loss.asnumpy(), sum(decode_lengths))
        top5accs.update(top5.asnumpy(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        # Print status
        if i % print_freq == 0:
            logger.info(f'Epoch:[{epoch}][{i}/{step_size_train}]\t'
                  f'Batch Time:{batch_time.val:.4f}s\t'
                  # f'Data Load Time:{data_time.val:.3f}s\t'
                  f'Loss:{losses.val.item():.4f} (Avg:{losses.avg.item():.4f})\t'
                  f'Top5 Acc:{top5accs.val:.3f} (Avg:{top5accs.avg.item():.4f})')


def validate(val_loader, encoder, decoder):
    '''
    Performs one epoch's validation.
    :param val_loader: dataloader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :return: BLEU-4 score
    '''
    decoder.set_train(False)  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.set_train(False)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    # explicitly disable gradient calculation to avoid CUDA memory error
    for i, (imgs, caps, caplens, allcaps) in tqdm(enumerate(val_loader), total=step_size_val):

        loss, scores, caps_sorted, decode_lengths, sort_ind = forward_fn(imgs, caps, caplens)
        # Keep track of metrics
        top5 = accuracy(scores, caps_sorted[:, 1:max(caplens)], caplens, 5)
        losses.update(loss.asnumpy(), sum(decode_lengths))
        top5accs.update(top5.asnumpy(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        # Print status
        if i % print_freq == 0:
            logger.info(f'Epoch:[{i}][{step_size_val}]\t'
                  f'Batch Time:{batch_time.val:.3f}s\t'
                  # f'Data Load Time:{data_time.val:.3f}s\t'
                  f'Loss:{losses.val.item():.4f} (Avg:{losses.avg.item():.4f})\t'
                  f'Top5 Acc:{top5accs.val:.4f} (Avg:{top5accs.avg.item():.4f})')

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        # References
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].asnumpy().tolist()
            img_caps = list(map(lambda c: [w for w in c if w not in {vocab.tokens_to_ids("<pad>"), vocab.tokens_to_ids("<start>")}],img_caps))
            references.append(img_caps)
        _, preds = ops.max(scores, axis=2)

        preds = preds.asnumpy().tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)
        assert len(references) == len(hypotheses)
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses, weights=[1])
        logger.info(
            f'\n * LOSS: {losses.avg:.3f}, TOP5 ACC: {top5accs.avg:.3f}, BLEU-4: {bleu4}\n')
        return bleu4





if __name__ == '__main__':

    # for i, (imgs, caps, caplens) in enumerate(data_loader):
    #     print(imgs.shape)
    #     print(caps.shape)
    #     print(caplens.shape)
    #     img = imgs
    #     cap = caps
    #     caplen = caplens
    #     break

    # output = encoder(img)
    # train_loop(data_loader, encoder, decoder, epoch=5)
    # validate(val_data_loader, encoder, decoder)
    main(epochs)
