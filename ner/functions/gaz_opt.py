__author__ = "liuwei"


import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

def get_batch_gaz(gazs, batch_size, max_seq_len, gpu=False):
    """
    rely on the gazs for batch_data, generation a batched gaz tensor for train
    Args:
        gazs: a list list list, gazs for batch_data
        batch_size: the size of batch
        max_seq_len: the max seq length
    """
    # we need guarantee that every word has the same number gaz, that is use paddding
    # record the really length
    gaz_seq_length = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    max_gaz_len = 1
    for i in range(batch_size):
        this_gaz_len = len(gazs[i])
        gaz_lens = [len(gazs[i][j]) for j in range(this_gaz_len)]
        gaz_seq_length[i, :this_gaz_len] = torch.LongTensor(gaz_lens)
        l = max(gaz_lens)
        if max_gaz_len < l:
            max_gaz_len = l

    # do padding
    gaz_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_gaz_len))).long()
    for i in range(batch_size):
        for j in range(len(gazs[i])):
            l = int(gaz_seq_length[i][j])
            gaz_seq_tensor[i, j, :l] = torch.LongTensor(gazs[i][j][:l])

    # get mask
    empty_tensor = (gaz_seq_length == 0).long()
    empty_tensor = empty_tensor * max_gaz_len
    gaz_seq_length = gaz_seq_length + empty_tensor

    gaz_mask_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_gaz_len))).long()
    for i in range(batch_size):
        for j in range(max_seq_len):
            l = int(gaz_seq_length[i][j])
            gaz_mask_tensor[i, j, :l] = 1
    del empty_tensor

    if gpu:
        gaz_seq_tensor = gaz_seq_tensor.cuda()
        gaz_seq_length = gaz_seq_length.cuda()
        gaz_mask_tensor = gaz_mask_tensor.cuda()

    return gaz_seq_tensor, gaz_seq_length, gaz_mask_tensor
