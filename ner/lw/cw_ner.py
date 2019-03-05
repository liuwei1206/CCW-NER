__author__ = "liuwei"

"""
A char word model
"""
import numpy as np
import torch
import torch.nn as nn

from ner.modules.gaz_embed import Gaz_Embed
from ner.modules.gaz_bilstm import Gaz_BiLSTM
from ner.model.crf import CRF
from ner.functions.utils import random_embedding
from ner.functions.gaz_opt import get_batch_gaz
from ner.functions.utils import reverse_padded_sequence

class CW_NER(torch.nn.Module):
    def __init__(self, data, type=1):
        print("Build char-word based NER Task...")
        super(CW_NER, self).__init__()

        self.gpu = data.HP_gpu
        label_size = data.label_alphabet_size
        self.type = type
        self.gaz_embed = Gaz_Embed(data, type)

        self.word_embedding = nn.Embedding(data.word_alphabet.size(), data.word_emb_dim)

        self.lstm = Gaz_BiLSTM(data, data.word_emb_dim + data.gaz_emb_dim, data.HP_hidden_dim)

        self.crf = CRF(data.label_alphabet_size, self.gpu)

        self.hidden2tag = nn.Linear(data.HP_hidden_dim * 2, data.label_alphabet_size + 2)

        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                random_embedding(data.word_alphabet_size, data.word_emb_dim)
            )

        if self.gpu:
            self.word_embedding = self.word_embedding.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def neg_log_likelihood_loss(self, gaz_list, reverse_gaz_list, word_inputs, word_seq_lengths, batch_label, mask):
        """
        get the neg_log_likelihood_loss
        Args:
            gaz_list: the batch data's gaz, for every chinese char
            reverse_gaz_list: the reverse list
            word_inputs: word input ids, [batch_size, seq_len]
            word_seq_lengths: [batch_size]
            batch_label: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        """
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        lengths = list(map(int, word_seq_lengths))

        # print('one ', reverse_gaz_list[0][:10])

        ## get batch gaz ids
        batch_gaz_ids, batch_gaz_length, batch_gaz_mask = get_batch_gaz(reverse_gaz_list, batch_size, seq_len, self.gpu)

        # print('two ', batch_gaz_ids[0][:10])

        reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask = get_batch_gaz(gaz_list, batch_size, seq_len, self.gpu)
        reverse_batch_gaz_ids = reverse_padded_sequence(reverse_batch_gaz_ids, lengths)
        reverse_batch_gaz_length = reverse_padded_sequence(reverse_batch_gaz_length, lengths)
        reverse_batch_gaz_mask = reverse_padded_sequence(reverse_batch_gaz_mask, lengths)

        ## word embedding
        word_embs = self.word_embedding(word_inputs)
        reverse_word_embs = reverse_padded_sequence(word_embs, lengths)

        ## gaz embedding
        gaz_embs = self.gaz_embed((batch_gaz_ids, batch_gaz_length, batch_gaz_mask))
        reverse_gaz_embs = self.gaz_embed((reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask))
        # print(gaz_embs[0][0][:20])

        ## lstm
        forward_inputs = torch.cat((word_embs, gaz_embs), dim=-1)
        backward_inputs = torch.cat((reverse_word_embs, reverse_gaz_embs), dim=-1)

        lstm_outs, _ = self.lstm((forward_inputs, backward_inputs), word_seq_lengths)

        ## hidden2tag
        outs = self.hidden2tag(lstm_outs)

        ## crf and loss
        loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return loss, tag_seq

    def forward(self, gaz_list, reverse_gaz_list, word_inputs, word_seq_lengths, mask):
        """
        Args:
            gaz_list: the forward gaz_list
            reverse_gaz_list: the backward gaz list
            word_inputs: word ids
            word_seq_lengths: each sentence length
            mask: sentence mask
        """
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        lengths = list(map(int, word_seq_lengths))

        ## get batch gaz ids
        batch_gaz_ids, batch_gaz_length, batch_gaz_mask = get_batch_gaz(reverse_gaz_list, batch_size, seq_len, self.gpu)
        reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask = get_batch_gaz(gaz_list, batch_size, seq_len, self.gpu)
        reverse_batch_gaz_ids = reverse_padded_sequence(reverse_batch_gaz_ids, lengths)
        reverse_batch_gaz_length = reverse_padded_sequence(reverse_batch_gaz_length, lengths)
        reverse_batch_gaz_mask = reverse_padded_sequence(reverse_batch_gaz_mask, lengths)

        ## word embedding
        word_embs = self.word_embedding(word_inputs)
        reverse_word_embs = reverse_padded_sequence(word_embs, lengths)

        ## gaz embedding
        gaz_embs = self.gaz_embed((batch_gaz_ids, batch_gaz_length, batch_gaz_mask))
        reverse_gaz_embs = self.gaz_embed((reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask))

        ## lstm
        forward_inputs = torch.cat((word_embs, gaz_embs), dim=-1)
        backward_inputs = torch.cat((reverse_word_embs, reverse_gaz_embs), dim=-1)

        lstm_outs, _ = self.lstm((forward_inputs, backward_inputs), word_seq_lengths)

        ## hidden2tag
        outs = self.hidden2tag(lstm_outs)

        ## crf and loss
        _, tag_seq = self.crf._viterbi_decode(outs, mask)

        return tag_seq
