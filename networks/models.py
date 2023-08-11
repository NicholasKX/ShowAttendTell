# -*- coding: utf-8 -*-
"""
Created on 2023/8/6 22:51 
@Author: Wu Kaixuan
@File  : models.py 
@Desc  : models 
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C

from networks.resnet import resnet101


class Encoder(nn.Cell):
    '''
    Encoder.
    '''

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.resnet = resnet101(pretrained=False)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(self.resnet.cells())[:-3]
        self.resnet = nn.SequentialCell(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def construct(self, images):

        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = ops.Transpose()(out, (0, 2, 3, 1))  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow
        """
        for p in self.resnet.get_parameters():
            p.requires_grad = False
        for c in list(self.resnet.cells())[5:]:
            for p in c.get_parameters():
                p.requires_grad = fine_tune


class Attention(nn.Cell):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        '''

        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        '''
        super(Attention, self).__init__()
        self.encoder_att = nn.Dense(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Dense(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Dense(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)  # softmax layer to calculate weights

    def construct(self, encoder_out, decoder_hidden):
        '''

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        '''
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(axis=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Cell):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        '''

        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        '''
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, has_bias=True)  # decoding LSTMCell
        self.init_h = nn.Dense(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Dense(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Dense(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Dense(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        '''
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        '''
        self.embedding.embedding_table.set_data(
            ms.Tensor(np.random.uniform(-0.1, 0.1, size=(self.vocab_size, self.embed_dim)).astype(np.float32)))
        self.fc.weight.set_data(
            ms.Tensor(np.random.uniform(-0.1, 0.1, size=(self.vocab_size, self.decoder_dim)).astype(np.float32)))
        self.fc.bias.set_data(ms.Tensor(np.zeros(self.vocab_size).astype(np.float32)))

    def load_pretrained_embeddings(self, embeddings):
        '''
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        '''
        self.embedding.embedding_table.set_data(ms.Tensor(embeddings))

    def fine_tune_embeddings(self, fine_tune=True):
        '''
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        '''
        for p in self.embedding.trainable_params():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        '''
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        '''
        mean_encoder_out = encoder_out.mean(axis=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def construct(self, encoder_out, encoded_captions, caption_lengths):
        '''

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_length: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        '''
        batch_size = encoder_out.shape[0]
        encoder_dim = encoder_out.shape[-1]
        vocab_size = self.vocab_size
        max_len = int(caption_lengths.max()) - 1

        # Flatten image
        encoder_out = encoder_out.view((batch_size, -1, encoder_dim))
        num_pixels = encoder_out.shape[1]

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(axis=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        decode_lengths = (caption_lengths - 1).asnumpy().tolist()

        prediction = ms.Tensor(np.zeros((batch_size, max_len, vocab_size)).astype(np.float32))
        alphas = ms.Tensor(np.zeros((batch_size, max_len, num_pixels)).astype(np.float32))

        for t in range(max_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(ms.ops.concat((embeddings[:batch_size_t, t, :], attention_weighted_encoding),axis=1),
                                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            prediction[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return prediction, alphas, encoded_captions, decode_lengths, sort_ind


class NetWithLossCell(nn.Cell):
    def __init__(self, encoder, decoder, loss_fn, alpha_c=1.):
        super(NetWithLossCell, self).__init__()
        self.encoder = encoder
        self.encoder.set_train(False)
        self.decoder = decoder
        self.decoder.set_train()
        self.loss_fn = loss_fn
        self.alpha_c = alpha_c

    def construct(self, img, encoded_captions, caption_lengths):
        img_out = self.encoder(img)
        scores, alphas = self.decoder(img_out, encoded_captions, caption_lengths)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        # Remove timesteps that we didn't decode at, or are pads
        targets = encoded_captions[:, 1:max(caption_lengths)]
        # Calculate loss
        loss = self.loss_fn(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1))
        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1. - alphas.sum(axis=1)) ** 2).mean()
        return loss


GRADIENT_CLIP_TYPE = 0
GRADIENT_CLIP_VALUE = 5.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """_clip_grad"""
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainingWrapper(nn.Cell):
    """TrainingWrapper"""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        # from dsj
        self.hyper_map = ms.ops.HyperMap()

    def construct(self, *args):
        """construct"""
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        # from dsj
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return F.depend(loss, self.optimizer(grads))


if __name__ == '__main__':
    # model = Encoder()
    # x = ms.Tensor(np.random.rand(1, 3, 224, 224).astype(np.float32))
    # out = model(x)
    # print(out.shape)

    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=200)
    x = ms.Tensor(np.random.rand(2, 3, 224, 224).astype(np.float32))
    out = encoder(x)
    print(out.shape)
    y = ms.Tensor(np.random.randint(0, 200, size=(1, 10)).astype(np.int32))
    y_1 = ms.Tensor(np.random.randint(0, 200, size=(1, 8)).astype(np.int32))

    # print(y_1)
    # print(y)
    prediction, encoded_captions, decode_lengths, alphas, sort_ind = decoder(out, y, ms.Tensor(np.array([10, 8])))
    print(prediction.shape)
    print(encoded_captions.shape)
    print(decode_lengths)
    print(alphas.shape)
    print(sort_ind)
