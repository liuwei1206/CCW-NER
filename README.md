An Encoding Strategy based Word-Character LSTM for Chinese NER
=============================================================
An model take both character and words as input for Chinese NER task.  
  
  
Models and results can be found at our NAACL 2019 paper "An Encoding Strategy based Word-Character LSTM for Chinese NER". It achieves state-of-the-art performance on most of the dataset.  


Most of the code is written with reference to Yang Jie's "NCRF++". To know more about "NCRF++", please refer to the paper "NCRF++: An Open-source Neural Sequence Labeling Toolkit".   


Requirement:
============================
Python 3.6  
Pytorch: 0.4.0  

if you want to use the tensorboard in our code, you should also install the followings:  
tensorboardX 1.2  
tensorflow 1.6.0


Input format:
=============================
CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.  
```cpp
美   B-LOC  
国   E-LOC  
的   O  
华   B-PER  
莱   I-PER  
士   E-PER  

我   O  
跟   O  
他   O  
谈   O  
笑   O  
风   O  
生   O   
```    

Pretrained Embeddings:
===============
Character embeddings: [gigword_chn.all.a2b.uni.ite50.vec](https://pan.baidu.com/s/1pLO6T9D)  
Word embeddings: [ctb.50d.vec](https://pan.baidu.com/s/1pLO6T9D)  


Run:
============
put each dataset to the data dir, and then simply run the .py file.   
For example, to run Weibo experiment, just run: python3 weibo.py

Cite:
========
@inproceedings{liu-etal-2019-encoding, \
    title = "An Encoding Strategy Based Word-Character {LSTM} for {C}hinese {NER}", \
    author = "Liu, Wei  and
      Xu, Tongge  and
      Xu, Qinghua  and
      Song, Jiayu  and
      Zu, Yueran", \
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies", \
    year = "2019", \
    publisher = "Association for Computational Linguistics", \
    url = "https://www.aclweb.org/anthology/N19-1247", \
    pages = "2379--2389" \
}

