# -*- coding: utf-8 -*-
"""
Created on 2023/8/11 23:29 
@Author: Wu Kaixuan
@File  : caption.py 
@Desc  : caption 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import mindspore.dataset.vision as vision
from PIL import Image
from mindspore.dataset import transforms
import mindspore as ms
from networks.models import Encoder, DecoderWithAttention, NetWithLossCell, TrainingWrapper
from dataset import FlickrDataset, build_vocabulary, Vocabulary


def caption_image_beam_search(encoder,
                              decoder,
                              image_path,
                              vocab,
                              beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(vocab)

    img = Image.open(image_path).convert("RGB")

    trans = transforms.Compose([
        vision.Resize((256, 256)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        vision.HWC2CHW()
    ])

    image = ms.Tensor(trans(img)[0]) # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
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

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = ms.ops.ones((k, 1, enc_image_size, enc_image_size))  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(ms.ops.concat((embeddings, awe), axis=1), (h, c))  # (s, decoder_dim)

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

        # Add new words to sequences, alphas
        seqs = ms.ops.concat((seqs[prev_word_inds], next_word_inds.unsqueeze(1)), axis=1)  # (s, step+1)
        seqs_alpha = ms.ops.concat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               axis=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != vocab.tokens_to_ids('<eos>')]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].asnumpy().tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].asnumpy().tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
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
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5).astype(int), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.asnumpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.asnumpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Generate Caption')

    parser.add_argument('--img', '-i', default="test_imgs/dog.jpg", help='path to image')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    vocab = Vocabulary("flickr8k/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json")
    vocab_size = len(vocab)

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    encoder_checkpoint = "checkpoint/ms_encoder.ckpt"  # 模型路径
    decoder_checkpoint = "checkpoint/ms_decoder.ckpt"  # 模型路径
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
        print("load encoder checkpoint success")
    if decoder_checkpoint is not None:
        decoder_param_dict = ms.load_checkpoint(decoder_checkpoint)
        param_not_load, _ = ms.load_param_into_net(decoder, decoder_param_dict)
        print("load decoder checkpoint success")

    encoder.set_train(False)
    decoder.set_train(False)


    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, vocab, args.beam_size)
    rev_word_map = vocab.idx2word
    alphas = ms.Tensor(alphas)
    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
