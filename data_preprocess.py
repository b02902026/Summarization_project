import sys, os
from copy import deepcopy

USE_TS = True


def indexing_word(article, title, word2idx, target2idx):
    ext_vocab = [{} for _ in range(len(article))]
    ext_count = [0 for _ in range(len(article))]
    article_tar = deepcopy(article)

    for sid,s in enumerate(article):
        for wid,w in enumerate(s):
            # index with target vocab or extent vocab
            if w in target2idx:
                article_tar[sid][wid] = target2idx[w]
            elif w in ext_vocab[sid]:
                article_tar[sid][wid] = ext_vocab[sid][w]
            else:
                ext_vocab[sid][w] = len(target2idx) + ext_count[sid]
                article_tar[sid][wid] = ext_vocab[sid][w]
                ext_count[sid] += 1

            # index with source vocab
            if w in word2idx:
                article[sid][wid] = word2idx[w]
            else:
                print(w)
                article[sid][wid] = word2idx['UNK']

    for sid,s in enumerate(title):
        for wid,w in enumerate(s):
            if w in target2idx:
                title[sid][wid] = target2idx[w]
            else:
                title[sid][wid] = target2idx['UNK']


    return [article, article_tar, title, ext_vocab, ext_count]

def make_word_idx_dict(word_freq, sw = None):
    word2idx = {'PAD':0, 'SOS':1, 'EOS':2}
    wc = 3
    for k,v in word_freq.items():
        if v > 0:
            word2idx[k] = wc
            wc += 1

    return word2idx

def read_textfile(DATA_DIR='sumdata/train'):
    #word2idx = {'PAD':0, 'SOS':1, 'EOS':2}
    word_freq_s = {}
    word_freq_t = {}
    article = []
    title = []
    wc = 3
    max_a, max_t = 0, 0
    thres = 1000
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
    target2idx = make_word_idx_dict(word_freq_t)
    #article, title, ext_vocab = indexing_word(article, title, word2idx, target2idx)

    return [article, title, word2idx, target2idx, (max_a, max_t)]


def padding(article, title, word2idx, maxl):
    tgt_l = []
    max_a, max_t = maxl
    #print([len(x) for x in article])
    for i, _  in enumerate(zip(article, title)):
        tgt_l.append(len(article[i]))
        article[i] += ['PAD' for _ in range(max_a - len(article[i]))]
        title[i] += ['PAD' for _ in range(max_t - len(title[i]))]

    #print(tgt_l)
    return [article, title, tgt_l]

