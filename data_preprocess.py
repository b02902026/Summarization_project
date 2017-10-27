import sys, os
from collections import OrderedDict
from copy import deepcopy
import operator

USE_TS = True

def read_cnn_dailymail(DATA_PATH='cnn/train.txt',thres=0):
    word_freq = {}
    article, target = [], []
    max_a, max_t = 0, 0
    with open(DATA_PATH) as f:
        count = 0
        instance_n = 0
        for line in f:
            words = line.strip().split()
            word_list = []
            if count % 4 == 0:
                for w in words:
                    if w not in word_freq:
                        word_freq[w] = 0
                    word_freq[w] += 1
                    word_list.append(w)
                max_t = max(max_t, len(word_list))
                word_list.append('EOS')
                target.append(word_list[:])

            elif count % 4 == 2:
                for w in words:
                    if w not in word_freq:
                        word_freq[w] = 0
                    word_freq[w] += 1
                    word_list.append(w)
                max_a = max(max_a, len(word_list))
                word_list.append('EOS')
                article.append(word_list[:])

                instance_n += 1
                if instance_n >= thres:
                    break
            count += 1

    word2idx = make_word_idx_dict(word_freq)
    target2idx = word2idx
    #target2idx = make_word_idx_dict(word_freq)
    #article, title, ext_vocab = indexing_word(article, title, word2idx, target2idx)

    return [article, target, word2idx, target2idx, (max_a, max_t)]



def make_word_idx_dict(word_freq, sw = None):
    word2idx = {'PAD':0, 'SOS':1, 'EOS':2, 'UNK':3}
    wc = 4
    word_freq_list = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
    for k,v in word_freq_list:
        if wc >= 50000:
            break
        word2idx[k] = wc
        wc += 1
    del word_freq

    return word2idx

def read_textfile(DATA_DIR='sumdata/train', thres=100):
    #word2idx = {'PAD':0, 'SOS':1, 'EOS':2}
    word_freq_s = {}
    word_freq_t = {}
    word_freq = {}
    article = []
    title = []
    max_a, max_t = 0, 0
    tc = 0
    with open(os.path.join(DATA_DIR,'train.article.txt'),'r') as fx:
        for line in fx:
            words = line.strip().split()
            wid_list = []
            for w in words:
                if w not in word_freq_s:
                    word_freq_s[w] = 0
                    #word2idx[w] = wc
                    #wc += 1
                word_freq_s[w] += 1
                wid_list.append(w)

            wid_list.append('EOS')
            max_a = len(wid_list) if len(wid_list) > max_a else max_a
            article.append(wid_list[:])
            if USE_TS:
                tc += 1
                if tc >= thres: break

    tc = 0
    with open(os.path.join(DATA_DIR,'train.title.txt'),'r') as fy:
        for line in fy:
            words = line.strip().split()
            wid_list = []
            for w in words:
                if w not in word_freq_t:
                    word_freq_t[w] = 0
                    #word2idx[w] = wc
                    #wc += 1
                word_freq_t[w] += 1
                wid_list.append(w)

            wid_list.append('EOS')
            max_t = len(wid_list) if len(wid_list) > max_t else max_t
            title.append(wid_list[:])
            if USE_TS:
                tc += 1
                if tc >= thres: break

    word2idx = make_word_idx_dict(word_freq_s)
    target2idx = word2idx
    #target2idx = make_word_idx_dict(word_freq_t)
    #article, title, ext_vocab = indexing_word(article, title, word2idx, target2idx)

    return [article, title, word2idx, target2idx, (max_a, max_t)]


