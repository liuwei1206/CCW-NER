# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-12 22:09:37
import sys
import numpy as np
from ner.utils.alphabet import Alphabet
NULLKEY = "-null-"
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def read_word_instance(input_file, word_alphabet, label_alphabet, number_normalized, max_sent_lengths):
    """
    read only word msg, no char.
    note that here word is chinese character!
    """
    instence_texts = []
    instence_ids = []
    entity_num = 0
    # max_sent_lengths = 1000
    with open(input_file, 'r', errors='ignore') as f:
        in_lines = f.readlines()
        words = []
        labels = []
        word_ids = []
        label_ids = []

        for line in in_lines:
            if len(line) > 2:
                # len less than 2 mean a blank line
                pairs = line.strip().split()
                word = pairs[0]
                if number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                if "B-" in label or "S-" in label:
                    entity_num += 1
                words.append(word)
                labels.append(label)
                word_ids.append(word_alphabet.get_index(word))
                label_ids.append(label_alphabet.get_index(label))
            else:
                if (max_sent_lengths < 0) or (len(words) < max_sent_lengths):
                    instence_texts.append([words, labels])
                    instence_ids.append([word_ids, label_ids])
                # else:
                #     print(len(words))
                #     print('so long!!!')
                words = []
                labels = []
                word_ids = []
                label_ids = []
        # if len(words) > 0:
        #     instence_texts.append([words, labels])
        #     instence_ids.append([word_ids, label_ids])

    # print("entity num: ", entity_num)
    return instence_texts, instence_ids


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized,max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    """
    read the word and char msg, all read into instance_texts and instance_ids

    note that, in the file, every line is a single chinese character and its tag,
    but when read into instace_texts and instance_ids, every line is a sentence character,
    and sentence character ids
    """
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, chars, labels])
                instence_Ids.append([word_Ids, char_Ids,label_Ids])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def read_seg_instance(input_file, word_alphabet, biword_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    """
        read the word, biword and  char msg, all read into instance_texts and instance_ids

        note that, in the file, every line is a single chinese character and its tag,
        but when read into instace_texts and instance_ids, every line is a sentence character,
        and sentence character ids
        """
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, biwords, chars, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids,label_Ids])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids

def read_instance_with_gaz_no_char(input_file, gaz, word_alphabet, biword_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, use_single):
    """
    read instance with, word, biword, gaz, lable, no char
    Args:
        input_file: the input file path
        gaz: the gaz obj
        word_alphabet: word
        biword_alphabet: biword
        gaz_alphabet: gaz
        label_alphabet: label
        number_normalized: true or false
        max_sent_length: the max length
    """
    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    labels = []
    word_Ids = []
    biword_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) - 1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):
                gazs = []
                gaz_Ids = []
                gazs_length = []
                w_length = len(words)

                reverse_gazs = [[] for i in range(w_length)]
                reverse_gaz_Ids = [[] for i in range(w_length)]
                flag = [0 for f in range(w_length)]
                # assign sub-sequence to every chinese letter
                for i in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[i:])

                    if use_single and len(matched_list) > 0:
                        f_len = len(matched_list[0])
                        
                        if (flag[i] == 1 or len(matched_list) > 1) and len(matched_list[-1]) == 1:
                            matched_list = matched_list[:-1]                    

                        for f_pos in range(i, i+f_len):
                            flag[f_pos] = 1                        

                    matched_length = [len(a) for a in matched_list]

                    gazs.append(matched_list)
                    matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_Id:
                        # gaz_Ids.append([matched_Id, matched_length])
                        gaz_Ids.append(matched_Id)
                        gazs_length.append(matched_length)
                    else:
                        gaz_Ids.append([])
                        gazs_length.append([])

                for i in range(w_length-1, -1, -1):
                    now_pos_gaz = gazs[i]
                    now_pos_gaz_Id = gaz_Ids[i]
                    now_pos_gaz_len = gazs_length[i]

                    ## Traversing it
                    l = len(now_pos_gaz)
                    assert len(now_pos_gaz) == len(now_pos_gaz_Id)
                    for j in range(l):
                        width = now_pos_gaz_len[j]
                        end_char_pos = i + width - 1

                        reverse_gazs[end_char_pos].append(now_pos_gaz[j])
                        reverse_gaz_Ids[end_char_pos].append(now_pos_gaz_Id[j])


                instence_texts.append([words, biwords, gazs, reverse_gazs, labels])
                instence_Ids.append([word_Ids, biword_Ids, gaz_Ids, reverse_gaz_Ids, label_Ids])
            words = []
            biwords = []
            labels = []
            word_Ids = []
            biword_Ids = []
            label_Ids = []
            gazs = []
            reverse_gazs = []
            gaz_Ids = []
            reverse_gaz_Ids = []
    return instence_texts, instence_Ids

def read_instance_with_gaz(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                gazs = []
                gaz_Ids = []
                w_length = len(words)
                # print sentence 
                # for w in words:
                #     print w," ",
                # print
                for idx in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    # print idx,"----------"
                    # print "forward...feed:","".join(words[idx:])
                    # for a in matched_list:
                    #     print a,len(a)," ",
                    # print

                    # print matched_length

                    gazs.append(matched_list)
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])
                    
                instence_texts.append([words, biwords, chars, gazs, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
            gazs = []
            gaz_Ids = []
    return instence_texts, instence_Ids


def read_instance_with_gaz_in_sentence(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    for idx in range(len(in_lines)):
        pair = in_lines[idx].strip()
        orig_words = list(pair[0])
        
        if (max_sent_length > 0) and (len(orig_words) > max_sent_length):
            continue
        biwords = []
        biword_Ids = []
        if number_normalized:
            words = []
            for word in orig_words:
                word = normalize_word(word)
                words.append(word)
        else:
            words = orig_words
        word_num = len(words)
        for idy in range(word_num):
            if idy < word_num - 1:
                biword = words[idy]+words[idy+1]
            else:
                biword = words[idy]+NULLKEY
            biwords.append(biword)
            biword_Ids.append(biword_alphabet.get_index(biword))
        word_Ids = [word_alphabet.get_index(word) for word in words]
        label = pair[-1]
        label_Id =  label_alphabet.get_index(label)
        gazs = []
        gaz_Ids = []
        word_num = len(words)
        chars = [[word] for word in words]
        char_Ids = [[char_alphabet.get_index(word)] for word in words]
        ## print sentence 
        # for w in words:
        #     print w," ",
        # print
        for idx in range(word_num):
            matched_list = gaz.enumerateMatchList(words[idx:])
            matched_length = [len(a) for a in matched_list]
            # print idx,"----------"
            # print "forward...feed:","".join(words[idx:])
            # for a in matched_list:
            #     print a,len(a)," ",
            # print
            # print matched_length
            gazs.append(matched_list)
            matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]
            if matched_Id:
                gaz_Ids.append([matched_Id, matched_length])
            else:
                gaz_Ids.append([])
        instence_texts.append([words, biwords, chars, gazs, label])
        instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Id])
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0

    ## we should also init the index 0
    pretrain_emb[0, :] = np.random.uniform(-scale, scale, [1, embedd_dim])

    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


       
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
