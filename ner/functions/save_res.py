__author__ = "liuwei"


"""
some util functions
"""

import torch
import numpy as np
import os

def save_gold_pred(instances_text, preds, golds, name):
    """
    save the gold and pred result to do compare
    Args:
        instances_text: is a list list, each list is a sentence
        preds: is also a list list, each list is a sentence predict tag
        golds: is also a list list, each list is a sentence gold tag
        name: train? dev? or test
    """
    sent_len = len(instances_text)

    assert len(instances_text) == len(preds)
    assert len(preds) == len(golds)

    dir = "data/result/resume/"
    file_path = os.path.join(dir, name)
    num = 1
    with open(file_path, 'w') as f:
        f.write("wrod   gold   pred\n")
        for sent, gold, pred in zip(instances_text, golds, preds):
            # for each sentence
            for word, w_g, w_p in zip(sent[0], gold, pred):
                if w_g != w_p:
                    f.write(word)
                    f.write("   ")
                    f.write(w_g)
                    f.write("   ")
                    f.write(w_p)
                    f.write("   ")
                    f.write(str(num))
                    f.write("\n")
                    num += 1
                else:
                    f.write(word)
                    f.write("   ")
                    f.write(w_g)
                    f.write("   ")
                    f.write(w_p)
                    f.write("\n")

            f.write("\n")


