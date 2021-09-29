'make predictions using complete input sub-sentence'
from .BasicModule import BasicModule
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
from models.WeightDrop import WeightDrop
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class combinedHidden(nn.Module):
    """calculate combined_hidden as h(e) = tanh(W_forward*h_forward + W_back*h_back + bias)
    :return  combined_hidden, size = hidden_size"""
    def __init__(self, input_dim, output_dim):
        super(combinedHidden, self).__init__()
        # Create the layer parameters.
        if torch.cuda.is_available():
            self.weight_forward = Parameter(torch.Tensor(input_dim, output_dim).cuda())
            self.weight_backward = Parameter(torch.Tensor(input_dim, output_dim).cuda())
            self.bias = Parameter(torch.Tensor(1, output_dim).cuda())
        else:
            self.weight_forward = Parameter(torch.Tensor(input_dim, output_dim))
            self.weight_backward = Parameter(torch.Tensor(input_dim, output_dim))
            self.bias = Parameter(torch.Tensor(1, output_dim))

        # intialize the weight and bias parameters using random values.
        self.weight_forward.data.uniform_(-0.001, 0.001)
        self.weight_backward.data.uniform_(-0.001, 0.001)
        self.bias.data.uniform_(-0.001, 0.001)

    def forward(self,h_forward, h_back):
        # calculate combined_hidden as h(e) = tanh(W_forward*h_forward + W_back*h_back + bias)
        batch_expanded_bias = self.bias.expand(h_forward.size(0), self.bias.size(1))  # B x H
        combined_hidden = torch.mm(h_forward,self.weight_forward) + torch.mm(h_back, self.weight_backward) + batch_expanded_bias
        return F.tanh(combined_hidden)


class SelfAttention(nn.Module):
    "implement attention module as in  https://arxiv.org/abs/1804.06659 "
    def __init__(self, attention_size, batch_first=True, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        if torch.cuda.is_available():
            self.attention_weights = Parameter(torch.FloatTensor(attention_size)).cuda()
        else:
            self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        self.attention_weights.data.uniform_(-0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths)
        mask = Variable(torch.ones(attentions.size())).detach()
        if torch.cuda.is_available():
            mask = mask.cuda()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):
        "perform dot product of the attention vector and each hidden state"
        # inputs (B X L X 2H), scores(BXL)
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        weighted_context = weighted.sum(1).squeeze()

        return weighted_context, scores

class StructuredSelfAttention(nn.Module):
    "implementation of the paper A Structured Self-Attentive Sentence Embedding"
    def __init__(self, hidden_size, d_a=128, r=3):
        super(StructuredSelfAttention, self).__init__()
        self.linear_first = torch.nn.Linear(hidden_size, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.r = r

    def get_masked_attn(self, inputs, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths)
        mask = Variable(torch.ones(inputs.size())).detach()
        if torch.cuda.is_available():
            mask = mask.cuda()

        if inputs.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        return mask*inputs

    def renormalize(self, inputs):
        "normalize input: size(BxLxR) on dim 1(L)"
        sums = inputs.sum(dim=1, keepdim=True)
        return inputs/sums

    def get_avg_attn(self, attention):
        "attention size  B X r X L"
        attn_sum = attention.sum(dim=1) # sum over r
        avg_attention = attn_sum.t() / attn_sum.sum(dim=1)  # (LXB)(sum over r) / B (sum over L)=> LxB
        return avg_attention.t() # BXL

    def forward(self, inputs, lengths):
        x = F.tanh(self.linear_first(inputs))
        x = self.linear_second(x)
        x = F.softmax(x, dim=1)  # B X L X r
        masked_x = self.renormalize(self.get_masked_attn(x,lengths))  # mask attention by lengths (*)
        attention = masked_x.transpose(1, 2)  # BxrxL
        ####uncomment to get avg embeddings earlier than avg attention
        # sentence_embeddings = attention @ inputs  # matmul(inputs:BXLX2H)
        # avg_sentence_embeddings = torch.sum(sentence_embeddings, dim=1) / self.r

        avg_attention = self.get_avg_attn(attention)
        avg_sentence_embeddings = (avg_attention.unsqueeze(1) @ inputs).squeeze(1) # BX2H(apply avg attention)
        return avg_sentence_embeddings, avg_attention # return context vector, attention



class AttnRNN(BasicModule):
    def __init__(self, dictionary, args):
        super(AttnRNN, self).__init__()
        self.model_name = 'AttnRNN'
        self.rnn_type = args.rnn_type
        self.apply_attn = args.apply_attn  # use self attention
        if self.apply_attn:
            self.attn_type = args.attn_type
        self.cnn_rnn = args.get("cnn_rnn")
        self.combine_hidden = args.combine_hidden

        self.s_size = len(dictionary['source'])
        self.out_size = len(dictionary['target'])
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.nlayers = args.nlayers  # number of layers
        self.drop_embed_prob = args.drop_embed_prob
        self.drop_rnn_prob = args.drop_rnn_prob
        self.drop_out_prob = args.drop_out_prob
        self.bidirectional = args.is_bidirectional
        self.variational_dropout_prob = args.variational_dropout_prob
        self.num_directions = int(self.bidirectional) + 1
        self.feature_dimension = 0
        self.fc_in_dim = args.hidden_size*self.num_directions
        self.use_pretrained_embed = args.use_pretrained_embed
        self.da = args.da  # hyperparameter for structured attn
        self.r = args.r    # hyperparameter for structured attn(number of hops for each sentence)
        self.follow_paper=args.follow_paper
        self.concat_out_emb = args.concat_out_emb


        # 1. initialize embedding
        if self.use_pretrained_embed and args.lang == 'de':
            pretrained_weights = self.get_embed_weights(dictionary)
            print("Pretrained embedding loaded!")
            self.word_embedding = nn.Embedding(self.s_size, self.embed_size)
            self.word_embedding.weight = nn.Parameter(pretrained_weights)
        else:
            self.word_embedding = nn.Embedding(self.s_size, self.embed_size, padding_idx=0)
        self.feature_dimension += self.embed_size


        # 2. initialize cnn if declared
        if self.cnn_rnn:
            self.filter_size = self.hidden_size
            self.kernel_size = 3
            self.conv_layers = nn.Sequential(nn.Conv1d(self.feature_dimension, self.filter_size, self.kernel_size),
                                             nn.Conv1d(self.filter_size, self.filter_size, self.kernel_size))
            self.feature_dimension = self.filter_size

        # 3. initialize rnn encoder
        if self.rnn_type == 'LSTM':
            self.encoder = nn.LSTM(self.feature_dimension,
                                   self.hidden_size,
                                   self.nlayers,
                                   dropout=self.drop_rnn_prob,
                                   bidirectional=self.bidirectional,
                                   batch_first=True)
        elif self.rnn_type == 'GRU':
            self.encoder = nn.GRU(self.feature_dimension,
                                  self.hidden_size,
                                  self.nlayers,
                                  dropout=self.drop_rnn_prob,
                                  bidirectional=self.bidirectional,
                                  batch_first=True)

        # 4. initialize dropout(embedding dropout, rnn dropout)
        self.drop_embed = nn.Dropout(self.drop_embed_prob)
        # setting up rnn dropout
        if self.variational_dropout_prob > 0:
            self.rnn = WeightDrop(self.encoder, ['weight_hh_l0'], dropout=self.variational_dropout_prob)
        else:
            self.rnn = self.encoder

        # 5. combine_hidden if declared, o.w. concatenate hidden
        if args.combine_hidden:
            # last hidden_layer h(e) = tanh(W_forward*h_forward + W_back*h_back + bias)
            self.cat_hidden = combinedHidden(self.hidden_size, self.hidden_size)
            self.fc_in_dim = self.hidden_size


        # 6. initialize attn layer if declared
        if self.apply_attn:
            if self.attn_type == 'self_attn':
                self.attn = SelfAttention(self.fc_in_dim)
            elif self.attn_type == 'structured_self_attn':
                if args.concat_out_emb:
                    self.fc_in_dim = self.fc_in_dim + self.feature_dimension
                self.attn = StructuredSelfAttention(self.fc_in_dim, self.da, self.r)
                if args.follow_paper:
                    self.fc = nn.Linear(self.fc_in_dim*self.r, self.out_size)

        # 7. initialize fc layer(if allow target, bilinear, o.w. linear)

        self.fc = nn.Linear(self.fc_in_dim, self.out_size)

        self.out = nn.Sequential(nn.BatchNorm1d(self.out_size),
                                 nn.Dropout(self.drop_out_prob))  # dropout after linear

    def forward(self, batch_seqs, lengths, hidden=None):
        # 1. embedding_layer
        embedded_word = self.word_embedding(batch_seqs['input'])
        embedded_source = self.drop_embed(embedded_word)  # B X L x H

        # 2. apply cnn before rnn if declared
        if self.cnn_rnn:
            "if use cnn_rnn, pass cnn de-inflected_full to rnn, o.w. use word embedding"
            self.conv_output = self.conv_layers(embedded_source.permute(0,2,1)).permute(0,2,1) # B x l' x filter_size
            outputs, hidden = self.rnn(self.conv_output, hidden)
        else:
            packed = pack_padded_sequence(embedded_source, lengths, batch_first=True)
            outputs, hidden = self.rnn(packed, hidden)  # hidden = 2*2 X B X H

        # 3. apply attention if declared, else use last hidden state
        if self.apply_attn:
            concated_output, l = pad_packed_sequence(outputs, batch_first=True)  # B X L X 2H
            if self.concat_out_emb:
                concated_out_emb = torch.cat([concated_output, embedded_source], dim=2) # B X L X(2H+E)
                context, scores = self.attn(concated_out_emb, l)  # weighted_context(B X 2H), scores(BXL)
            else:
                context, scores = self.attn(concated_output, l)  # weighted_context(B X 2H), scores(BXL)

            if self.follow_paper:
                context = context.view(batch_seqs['input'].size(0), -1)
        elif self.combine_hidden:
            if self.rnn_type == 'LSTM':
                context = self.cat_hidden(hidden[0][-2], hidden[0][-1]).contiguous() # get hidden state if LSTM
            else:
                context = self.cat_hidden(hidden[-2], hidden[-1]).contiguous()  # combine forward and back hidden state
        else:
            # concat backforward and forward last hidden
            if self.rnn_type == 'LSTM':
                context = torch.cat([hidden[0][-2], hidden[0][-1]], dim=1)
            else:
                context = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # 4. predict with target score if declared

        fc_out = self.fc(context)  # use context vector for prediction

        prediction = self.out(fc_out)  # batchnorm and dropout

        # return attention scores if use attention
        if self.apply_attn:
            return prediction, scores, hidden # prediction(B X OUT), scores(BXL)
        else:
            return prediction, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            if torch.cuda.is_available():
                return (Variable(weight.new(self.nlayers*self.num_directions, batch_size, self.hidden_size).zero_()).cuda(),
                        Variable(weight.new(self.nlayers*self.num_directions, batch_size, self.hidden_size).zero_()).cuda())
            else:
                return (Variable(weight.new(self.nlayers*self.num_directions, batch_size, self.hidden_size).zero_()),
                        Variable(weight.new(self.nlayers*self.num_directions, batch_size, self.hidden_size).zero_()))
        elif self.rnn_type == 'GRU':
            if torch.cuda.is_available():
                return Variable(weight.new(self.nlayers*self.num_directions, batch_size, self.hidden_size).zero_()).cuda()
            else:
                return Variable(weight.new(self.nlayers*self.num_directions, batch_size, self.hidden_size).zero_())
                        # Nlayes*Direction x B x H

    def get_embed_weights(self, dictionary):
        self.embed_size = 300
        pretrained_weight = torch.zeros(self.s_size, self.embed_size)
        for w,v in dictionary['source'].items():
            pretrained_weight[v] = torch.from_numpy(dictionary['embed'][w])
        return pretrained_weight

    # Regularization
    def l2_matrix_norm(self, m):
        """
        Frobenius norm calculation
        Args:
           m: {Variable} ||AAT - I||
        Returns:
            regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, dim=1), dim=1) ** 0.5).type(torch.DoubleTensor)