# -*- coding: utf-8 -*-
"""
Created on 2023/8/11 16:57 
@Author: Wu Kaixuan
@File  : evaluation.py
@Desc  : evaluation
"""

from typing import Literal
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype, ms_function
from tqdm import tqdm
from networks.models import Encoder, DecoderWithAttention, NetWithLossCell, TrainingWrapper
from dataset import FlickrDataset, build_vocabulary, Vocabulary
from mindspore.dataset import GeneratorDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu


def create_dataset_flickr(data,
                          vocab,
                          split: Literal["train", "val", "test"] = "train",
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


# Parameters
word_map_file = 'flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
encoder_checkpoint = "model_saved/encoder_epoch_3_bleu4_0.3718.ckpt"  # 模型路径
decoder_checkpoint = "model_saved/decoder_epoch_3_bleu4_0.3718.ckpt"  # 模型路径

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

data_dir = "flickr8k"
image_dir = "flickr8k/Images"
# image_dir = r"F:\Pytorch\ImageCaptioning\flickr8k\Images"
vocab = Vocabulary(word_map_file)
vocab_size = len(vocab)

# Load model
encoder = Encoder()
decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=vocab_size,
                               dropout=dropout)

if encoder_checkpoint is not None:
    encoder_param_dict = ms.load_checkpoint(encoder_checkpoint)
    param_not_load, _ = ms.load_param_into_net(encoder, encoder_param_dict)
    print(f"load encoder checkpoint from {encoder_checkpoint} success")
if decoder_checkpoint is not None:
    decoder_param_dict = ms.load_checkpoint(decoder_checkpoint)
    param_not_load, _ = ms.load_param_into_net(decoder, decoder_param_dict)
    print(f"load decoder checkpoint from {decoder_checkpoint} success")

encoder.set_train(False)
decoder.set_train(False)

# DataLoader
test_dataset = FlickrDataset(data_dir=r"flickr8k",
                             img_dir=image_dir,
                             split="test", vocab=vocab)

test_data = create_dataset_flickr(test_dataset, vocab, split="test", batch_size=1)
step_size_test = test_data.get_dataset_size()
test_data_loader = test_data.create_tuple_iterator()


def evaluate(beam_size, dataloader):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(dataloader, desc="EVALUATING AT BEAM SIZE " + str(beam_size), total=step_size_test)):

        k = beam_size

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.shape[1]
        encoder_dim = encoder_out.shape[-1]
        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.shape[1]

        # We'll treat the problem as having a batch size of k

        encoder_out = encoder_out.broadcast_to((k, num_pixels, encoder_dim))  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = ms.Tensor([[vocab.tokens_to_ids("<sos>")]] * k)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = ms.ops.zeros((k, 1))  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = decoder.decode_step(ms.ops.concat([embeddings, awe], axis=1), (h, c))  # (s, decoder_dim)
            scores = decoder.fc(h)  # (s, vocab_size)
            scores = ms.ops.log_softmax(scores, axis=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = (top_k_words / vocab_size).astype(ms.int64)  # (s)
            next_word_inds = (top_k_words % vocab_size).astype(ms.int64)  # (s)

            # Add new words to sequences
            seqs = ms.ops.concat((seqs[prev_word_inds], next_word_inds.unsqueeze(1)), axis=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab.tokens_to_ids('<eos>')]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].asnumpy().tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].asnumpy().tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {vocab.tokens_to_ids("<sos>"),
                                                     vocab.tokens_to_ids("<eos>"),
                                                     vocab.tokens_to_ids("<pad>")}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {vocab.tokens_to_ids("<sos>"),
                                                       vocab.tokens_to_ids("<eos>"),
                                                       vocab.tokens_to_ids("<pad>")}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4


if __name__ == '__main__':
    print(evaluate(beam_size=3, dataloader=test_data_loader))

