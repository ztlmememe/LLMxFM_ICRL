# from yelp/models.py in https://github.com/alvinchangw/CARA_EMNLP2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init

import json
import os
import numpy as np

from math import sqrt

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var
    
class MLP_Classify(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_Classify, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization in first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = F.sigmoid(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq2Decoder(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,
                 share_decoder_emb=False, hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq2Decoder, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder1 = nn.Embedding(ntokens, emsize)
        self.embedding_decoder2 = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder1 = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.decoder2 = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

        if share_decoder_emb:
            self.embedding_decoder2.weight = self.embedding_decoder1.weight

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder1.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder2.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder1.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder2.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, whichdecoder, indices, lengths, noise=False, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        if encode_only:
            return hidden
        decoded = self.decode(whichdecoder, hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state

        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of value 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_r)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, whichdecoder, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if whichdecoder == 1:
            embeddings = self.embedding_decoder1(indices)
        else:
            embeddings = self.embedding_decoder2(indices)

        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        if whichdecoder == 1:
            packed_output, state = self.decoder1(packed_embeddings, state)
        else:
            packed_output, state = self.decoder2(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, whichdecoder, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.resize_(batch_size, 1) 
        self.start_symbols.fill_(1)
        self.start_symbols = to_gpu(self.gpu, self.start_symbols)

        if whichdecoder == 1:
            embedding = self.embedding_decoder1(self.start_symbols)
        else:
            embedding = self.embedding_decoder2(self.start_symbols)

        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            if whichdecoder == 1:
                output, state = self.decoder1(inputs, state)
            else:
                output, state = self.decoder2(inputs, state)
            overvocab = self.linear(output.squeeze(1))
            
            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                assert 1 == 0
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            if whichdecoder == 1:
                embedding = self.embedding_decoder1(indices)
            else:
                embedding = self.embedding_decoder2(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices

class Seq2Seq2CNNDecoder(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, conv_windows="5-5-3", conv_strides="2-2-2",
                 conv_layer="500-700-1000", activation=nn.LeakyReLU(0.2, inplace=True), noise_r=0.2,
                 share_decoder_emb=False, hidden_init=False, dropout=0, gpu=False, pooling_enc="avg"):
        super(Seq2Seq2CNNDecoder, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        # for CNN encoder
        self.arch_conv_filters = conv_layer
        self.arch_conv_strides = conv_strides
        self.arch_conv_windows = conv_windows

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder1 = nn.Embedding(ntokens, emsize)
        self.embedding_decoder2 = nn.Embedding(ntokens, emsize)

        # for CNN encoder
        conv_layer_sizes = [emsize] + [int(x) for x in conv_layer.split('-')]
        conv_strides_sizes = [int(x) for x in conv_strides.split('-')]
        conv_windows_sizes = [int(x) for x in conv_windows.split('-')]
        self.encoder = nn.Sequential()

        for i in range(len(conv_layer_sizes) - 1):
            layer = nn.Conv1d(conv_layer_sizes[i], conv_layer_sizes[i + 1], \
                              conv_windows_sizes[i], stride=conv_strides_sizes[i])
            self.encoder.add_module("layer-" + str(i + 1), layer)

            bn = nn.BatchNorm1d(conv_layer_sizes[i + 1])
            self.encoder.add_module("bn-" + str(i + 1), bn)

            self.encoder.add_module("activation-" + str(i + 1), activation)

        if pooling_enc == "max":
            self.pooling_enc = nn.AdaptiveMaxPool1d(1)
        else:
            self.pooling_enc = nn.AdaptiveAvgPool1d(1)
        self.linear_enc = nn.Linear(1000, emsize)

        decoder_input_size = emsize+nhidden
        self.decoder1 = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.decoder2 = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

        if share_decoder_emb:
            self.embedding_decoder2.weight = self.embedding_decoder1.weight

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder1.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder2.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder1.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder2.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

        self.linear_enc.weight.data.uniform_(-initrange, initrange)
        self.linear_enc.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, whichdecoder, indices, lengths, noise=False, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        if encode_only:
            return hidden

        decoded = self.decode(whichdecoder, hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        embeddings = embeddings.transpose(1,2)
        c_pre_lin = self.encoder(embeddings)
        c_pre_lin = self.pooling_enc(c_pre_lin)
        c_pre_lin = c_pre_lin.squeeze(2)
        hidden = self.linear_enc(c_pre_lin)
        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        if norms.ndimension()==1:
            norms=norms.unsqueeze(1)
        hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_r)
            if self.gpu:
                gauss_noise = gauss_noise.cuda()
                
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, whichdecoder, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if whichdecoder == 1:
            embeddings = self.embedding_decoder1(indices)
        else:
            embeddings = self.embedding_decoder2(indices)

        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        if whichdecoder == 1:
            packed_output, state = self.decoder1(packed_embeddings, state)
        else:
            packed_output, state = self.decoder2(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, whichdecoder, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.resize_(batch_size, 1) 
        self.start_symbols.fill_(1) 
        self.start_symbols = to_gpu(self.gpu, self.start_symbols)

        if whichdecoder == 1:
            embedding = self.embedding_decoder1(self.start_symbols)
        else:
            embedding = self.embedding_decoder2(self.start_symbols)

        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            if whichdecoder == 1:
                output, state = self.decoder1(inputs, state)
            else:
                output, state = self.decoder2(inputs, state)
            overvocab = self.linear(output.squeeze(1))
            
            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                assert 1 == 0
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            if whichdecoder == 1:
                embedding = self.embedding_decoder1(indices)
            else:
                embedding = self.embedding_decoder2(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False, weight_init="default"):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        if weight_init == "kaiming_uniform":
            self.init_weights_kaiming_uniform()
        else:
            self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = torch.mean(x)

        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    # Initialize weights using He initialization
    def init_weights_kaiming_uniform(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)


        # for layer in self.layers:
        #     try:
        #         layer.weight.data.normal_(0, init_std)
        #         layer.bias.data.fill_(0)
        #     except:
        #         pass




class ReprogrammingEnc(nn.Module):
    def __init__(self, d_model, # d_model is the dimension of the input FM representation
                 n_heads, 
                 llm_model, 
                 num_prototypes, 
                 d_keys=None, # dim of keys in each head
                 d_llm=None, # dim of llm embeddings
                 attention_dropout=0.1,
                 output_projection=True):
        super(ReprogrammingEnc, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        # self.llm_model = llm_model tag: changed by ztl
        self.num_prototypes = num_prototypes
        self.output_projection = output_projection

        self.word_embeddings = llm_model.get_input_embeddings().weight


        self.vocab_size = self.word_embeddings.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        if self.output_projection:
            self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
            self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        else:
            self.value_projection = nn.Identity()
            self.out_projection = nn.Identity()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, enc_in):

            # Convert source_embeddings to the same dtype as enc_in
        if self.word_embeddings.dtype != enc_in.dtype: # tag: changed by ztl
            self.word_embeddings = torch.nn.Parameter(self.word_embeddings.to(enc_in.dtype))

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_in, source_embeddings, source_embeddings)

        return enc_out

    def reprogramming_layer(self, target_embedding, source_embedding, value_embedding):
        # Check the number of dim of target_embedding. If it is 2, this means that it is missing L, add one more dim for L
        if len(target_embedding.shape) == 2:
            target_embedding = target_embedding.unsqueeze(1)
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        if self.output_projection:
            value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        else:
            value_embedding = self.value_projection(value_embedding).unsqueeze(dim=1).expand(-1, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        enc_out = self.out_projection(out)

        return enc_out

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        if not self.output_projection:
            reprogramming_embedding = torch.mean(reprogramming_embedding, dim=-2) # average over heads

        return reprogramming_embedding


class MLP_Linear_K(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(), gpu=False,k=1):
        super(MLP_Linear_K, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.k = k

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

        layer = nn.Linear(layer_sizes[-1], k*self.noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # print(x.shape) 
        # torch.Size([1, 8192])
        x = x.view(self.k, self.noutput)  # Reshape to (K, noutput)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_Linear_init(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_Linear_init, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print("x Before: ", i, " ", x.shape)

            x = layer(x)
            # print("x After: ", i, " ", x.shape)
        return x

    def init_weights(self, method='xavier'):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_normal_(layer.weight)
                elif method == 'lecun':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
                elif method == 'uniform':
                    layer.weight.data.uniform_(-0.02, 0.02)
                elif method == 'sparse':
                    nn.init.sparse_(layer.weight, sparsity=0.1, std=0.02)

                elif method == 'dirac':
                    nn.init.dirac_(layer.weight)



                else:
                    layer.weight.data.normal_(0, 0.02)
                layer.bias.data.fill_(0)


import torch
import torch.nn as nn


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MLP_Nonlinear(nn.Module):
    def __init__(self, ninput, noutput=None, expansion_ratio=1):
        """
        A nonlinear MLP projector with RMSNorm, gate+up projection, and residual connection.
        Inspired by Llama's MLP structure.
        """
        super().__init__()
        if noutput is None:
            noutput = ninput
        self.ninput = ninput
        self.noutput = noutput
        self.expansion_ratio = expansion_ratio

        self.layer_norm = LlamaRMSNorm(ninput)
        self.input_proj = nn.Linear(ninput, noutput, bias=False)

        self.gate_proj = nn.Linear(noutput, noutput * expansion_ratio, bias=False)
        self.up_proj = nn.Linear(noutput, noutput * expansion_ratio, bias=False)
        self.down_proj = nn.Linear(noutput * expansion_ratio, noutput, bias=False)

        self.act_fn = nn.SiLU()

        self.init_weights()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.input_proj(x)
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden = self.down_proj(hidden)
        return hidden + x  # residual connection

    def init_weights(self):
        init_std = 0.02
        for layer in [self.input_proj, self.gate_proj, self.up_proj, self.down_proj]:
            layer.weight.data.normal_(0, init_std)
            if layer.bias is not None:
                layer.bias.data.zero_()



class Vicl_Linear(nn.Module):
    def __init__(self, in_features, out_features=None, dtype=torch.float32):
        """
        A simple linear projection layer, optionally projecting to a different dimensionality.
        """
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        self.init_weights()

    def forward(self, x):
        return self.linear(x)

    def init_weights(self):
        init_std = 0.02
        self.linear.weight.data.normal_(0, init_std)

class MLP_Linear(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_Linear, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print("x Before: ", i, " ", x.shape)

            x = layer(x)
            # print("x After: ", i, " ", x.shape)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_Mapper(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_Mapper, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print("x Before: ", i, " ", x.shape)

            x = layer(x)
            # print("x After: ", i, " ", x.shape)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass



class MLP_MultiHead(nn.Module):
    def __init__(self, ninput, noutput, layers, num_heads=8, activation=nn.ReLU(), output_length=10):
        super(MLP_MultiHead, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.num_heads = num_heads
        self.head_dim = noutput // num_heads
        self.output_length = output_length  # Define N for output shape (N, noutput)
        
        assert noutput % num_heads == 0, "Output dimension must be divisible by number of heads"
        
        # Define intermediate layers
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

        # Multi-head projections for output (N, 8192)
        self.head_projections = nn.ModuleList([
            nn.Linear(layer_sizes[-1], self.head_dim) for _ in range(num_heads)
        ])
        
        # Final linear transformation to get the required output length N
        self.output_projection = nn.Linear(self.head_dim * num_heads, noutput)

        self.init_weights()

    def forward(self, x):
        # Forward through intermediate layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = nn.ReLU()(x)  # Apply activation

        # Expand input to desired output length (N, feature_dim) before applying multi-head projections
        x_expanded = x.repeat(self.output_length, 1)  # Shape: (N, feature_dim)

        # Apply multi-head projections
        head_outputs = [head_proj(x_expanded) for head_proj in self.head_projections]
        multihead_output = torch.cat(head_outputs, dim=-1)  # Shape: (N, head_dim * num_heads)

        # Final linear transformation to get the output shape (N, noutput)
        output = self.output_projection(multihead_output)
        
        return output

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
        for head_proj in self.head_projections:
            head_proj.weight.data.normal_(0, init_std)
            head_proj.bias.data.fill_(0)

            
class MLP_Mapper_withoutbn_GELU(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.GELU(), gpu=False):
        super(MLP_Mapper_withoutbn_GELU, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print("x Before: ", i, " ", x.shape)

            x = layer(x)
            # print("x After: ", i, " ", x.shape)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

class MLP_Mapper_withoutbn(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_Mapper_withoutbn, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            # print("x Before: ", i, " ", x.shape)

            x = layer(x)
            # print("x After: ", i, " ", x.shape)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

            # # Add projection layer with linear layers and GELU activation
            # projection_layers = []
            # last_dim = backbone_output_dim
            # for i in range(self.ts_backbone_config['chronos_model']['projection_hidden_layer']):
            #     projection_layers.append(
            #         nn.Linear(last_dim, self.ts_backbone_config['chronos_model']["projection_hidden_dim"][i]))
            #     projection_layers.append(nn.GELU())
            #     last_dim = self.ts_backbone_config['chronos_model']["projection_hidden_dim"][i]

            # projection_layers.append(nn.Linear(last_dim, self.config.hidden_size))
            # self.ts_proj = nn.Sequential(*projection_layers)
            # logger.warning(
            #     f"Each layer with {self.ts_backbone_config['chronos_model']['projection_hidden_dim']} hidden units.")


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_r)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.resize_(batch_size, 1) 
        self.start_symbols.fill_(1) 

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


def load_models(load_path, epoch, twodecoders=False):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    if not twodecoders:
        autoencoder = Seq2Seq(emsize=model_args['emsize'],
                              nhidden=model_args['nhidden'],
                              ntokens=model_args['ntokens'],
                              nlayers=model_args['nlayers'],
                              hidden_init=model_args['hidden_init'])
    else:
        autoencoder = Seq2Seq2Decoder(emsize=model_args['emsize'],
                              nhidden=model_args['nhidden'],
                              ntokens=model_args['ntokens'],
                              nlayers=model_args['nlayers'],
                              hidden_init=model_args['hidden_init'])

    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model_{}.pt".format(epoch))
    gen_path = os.path.join(load_path, "gan_gen_model_{}.pt".format(epoch))
    disc_path = os.path.join(load_path, "gan_disc_model_{}.pt".format(epoch))

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise = Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise = Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences
