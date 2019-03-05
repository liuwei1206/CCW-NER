__author__ = "liuwei"

"""
the strategy to obtain gaz embedding. this embedding is concat to the char embedding
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ner.modules.highway import Highway
from ner.functions.utils import random_embedding
from ner.functions.utils import reverse_padded_sequence
from ner.functions.iniatlize import init_cnn_weight
from ner.functions.iniatlize import init_maxtrix_weight
from ner.functions.iniatlize import init_vector

class Gaz_Embed(torch.nn.Module):
    def __init__(self, data, type=1):
        """
        Args:
            data: all the data information
            type: the type of strategy, 1 for avg, 2 for short first, 3 for long first
        """
        print('build gaz embedding...')

        super(Gaz_Embed, self).__init__()

        self.gpu = data.HP_gpu
        self.data = data
        self.type = type
        self.gaz_dim = data.gaz_emb_dim
        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), data.gaz_emb_dim)
        self.dropout = nn.Dropout(p=0.5)

        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(
                torch.from_numpy(random_embedding(data.gaz_alphabet.size(), data.gaz_emb_dim))
            )

        self.filters = [[1, 20], [2, 30]]
        if self.type == 4:
            # use conv, so we need to define some conv
            # here we use 20 1-d conv, and 30 2-d conv
            self.build_cnn(self.filters)

            ## also use highway, 2 layers highway
            # self.highway = Highway(self.gaz_dim, num_layers=2)
            # if self.gpu:
            #     self.highway = self.highway.cuda()

        if self.type == 5:
            # use self-attention
            self.build_attention()

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()

    def build_cnn(self, filters):
        """
        build cnn for convolution the gaz embeddings
        Args:
            filters: the filters definetion
        """
        for filter in filters:
            k_size = filter[0]
            channel_size = filter[1]

            conv = torch.nn.Conv1d(
                in_channels=self.gaz_dim,
                out_channels=channel_size,
                kernel_size=k_size,
                bias=True
            )

            if self.gpu:
                conv = conv.cuda()

            init_cnn_weight(conv)
            self.add_module('conv_{}d'.format(k_size), conv)
            del conv

    def build_attention(self):
        """
        build a self-attention to weight add the values
        Args:
            max_gaz_num: the max gaz number
        """
        w1 = torch.zeros(self.gaz_dim, self.gaz_dim)
        w2 = torch.zeros(self.gaz_dim)

        if self.gpu:
            w1 = w1.cuda()
            w2 = w2.cuda()
        init_maxtrix_weight(w1)
        init_vector(w2)

        self.W1 = nn.Parameter(w1)
        self.W2 = nn.Parameter(w2)

    def forward(self, inputs):
        """
        the inputs is a tuple, include gaz_seq_tensor, gaz_seq_length, gaz_mask_tensor
        Args:
            gaz_seq_tensor: [batch_size, seq_len, gaz_num]
            gaz_seq_length: [batch_size, seq_len]
            gaz_mask_tensor: [batch_size, seq_len, gaz_num]
        """
        gaz_seq_tensor, gaz_seq_lengths, gaz_mask_tensor = inputs
        batch_size = gaz_seq_tensor.size(0)
        seq_len = gaz_seq_tensor.size(1)
        gaz_num = gaz_seq_tensor.size(2)

        # type = 1, short first; type = 2, long first; type = 3, avg; type = 4, cnn
        if self.type == 1:
            # short first
            gaz_ids = gaz_seq_tensor[:, :, 0]
            gaz_ids = gaz_ids.view(batch_size, seq_len, -1)
            gaz_ids = torch.squeeze(gaz_ids, dim=-1)
            if self.gpu:
                gaz_ids = gaz_ids.cuda()

            gaz_embs = self.gaz_embedding(gaz_ids)

            return gaz_embs

        elif self.type == 2:
            # long first
            select_ids = gaz_seq_lengths
            select_ids = select_ids.view(batch_size, seq_len, -1)

            # the max index = len - 1
            select_ids = select_ids - 1
            # print(select_ids[0])
            gaz_ids = torch.gather(gaz_seq_tensor, dim=2, index=select_ids)
            gaz_ids = gaz_ids.view(batch_size, seq_len, -1)
            gaz_ids = torch.squeeze(gaz_ids, dim=-1)
            if self.gpu:
                gaz_ids = gaz_ids.cuda()

            gaz_embs = self.gaz_embedding(gaz_ids)

            return gaz_embs

        elif self.type == 3:
            ## avg first
            # [batch_size, seq_len, gaz_num, gaz_dim]
            if self.gpu:
                gaz_seq_tensor = gaz_seq_tensor.cuda()

            gaz_embs = self.gaz_embedding(gaz_seq_tensor)

            # use mask to do select sum, mask: [batch_size, seq_len, gaz_num]
            pad_mask = gaz_mask_tensor.view(batch_size, seq_len, gaz_num, -1).float()

            # padding embedding transform to 0
            gaz_embs = gaz_embs * pad_mask
            
            # do sum at the gaz_num axis, result is [batch_size, seq_len, gaz_dim]
            gaz_embs = torch.sum(gaz_embs, dim=2)
            gaz_seq_lengths = gaz_seq_lengths.view(batch_size, seq_len, -1).float()
            gaz_embs = gaz_embs / gaz_seq_lengths

            return gaz_embs

        elif self.type == 4:
            ## use convolution
            # first get all the gaz embedding representation
            # [batch_size, seq_len, gaz_num, gaz_dim]
            input_embs = self.gaz_embedding(gaz_seq_tensor)

            # transform to [batch_size * seq_len, gaz_num, gaz_dim] to use Conv1d
            input_embs = input_embs.view(-1, gaz_num, self.gaz_dim)
            input_embs = torch.transpose(input_embs, 2, 1)
            input_embs = self.dropout(input_embs)

            gaz_embs = []

            for filter in self.filters:
                k_size = filter[0]

                conv = getattr(self, 'conv_{}d'.format(k_size))
                convolved = conv(input_embs)

                # convolved is [batch_size * seq_len, channel, width-k_size+1]
                # do active and max-pool, [batch_size * seq_len, channel]
                convolved, _ = torch.max(convolved, dim=-1)
                if True:
                    convolved = F.tanh(convolved)

                gaz_embs.append(convolved)

            # transpose to [batch_size, seq_len, gaz_dim]
            gaz_embs = torch.cat(gaz_embs, dim=-1)

            # gaz_embs = self.dropout(gaz_embs)
            # gaz_embs = self.highway(gaz_embs)
            gaz_embs = gaz_embs.view(batch_size, seq_len, -1)

            return gaz_embs

        elif self.type == 5:
            ## self attention
            gaz_embs = self.gaz_embedding(gaz_seq_tensor)
            # print('origin: ', gaz_embs[0][0][:20])
            input_embs = gaz_embs.view(-1, gaz_num, self.gaz_dim)
            input_embs = torch.transpose(input_embs, 2, 1)

            ### step1, cal alpha
            # cal the alpha, result is [batch * seq_len, d1, gaz_num]
            alpha = torch.matmul(self.W1, input_embs)
            alpha = F.tanh(alpha)

            # result is [batch * seq_len, gaz_num]
            alpha = torch.transpose(alpha, 2, 1)
            weight2 = self.W2
            alpha = torch.matmul(alpha, weight2)

            # before softmax, we need to mask
            alpha = alpha * gaz_mask_tensor.contiguous().view(-1, gaz_num).float()
            # zero_mask = (1 - gaz_mask_tensor.float()) * (-2**31 + 1)
            # zero_mask = zero_mask.contiguous().view(-1, gaz_num)
            # alpha = alpha + zero_mask

            ### step2 do softmax,
            # [batch * seq_len, gaz_num]
            # alpha = torch.exp(alpha)
            # total_alpha = torch.sum(alpha, dim=-1, keepdim=True)
            # alpha = torch.div(alpha, total_alpha)
            alpha = F.softmax(alpha, dim=-1)
            alpha = alpha.view(batch_size, seq_len, gaz_num, -1)

            ### step3, weighted add, [batch_size, seq_len, gaz_num, gaz_dim]
            gaz_embs = gaz_embs * alpha
            gaz_embs = torch.sum(gaz_embs, dim=2)

            return gaz_embs        




