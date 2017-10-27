from copy import deepcopy
import torch
from torch.autograd import Variable
import numpy as np
def indexing(source, target, source_vocab, target_vocab):

    for sid, sentence in enumerate(source):
        for wid, word in enumerate(sentence):
            if word in source_vocab:
                source[sid][wid] = source_vocab[word]
            else:
                source[sid][wid] = source_vocab['UNK']

    for sid, sentence in enumerate(target):
        for wid, word in enumerate(sentence):
            if word in target_vocab:
                target[sid][wid] = target_vocab[word]
            else:
                target[sid][wid] = target_vocab['UNK']

    return [source, target]

def batch_index(article, title, word2idx, target2idx):
    article_b = deepcopy(article)
    title_b = deepcopy(title)
    article_b, extend_article_b, title_b, extend_vocab_b, extend_count_b = \
        index_and_extend(article_b, title_b, word2idx, target2idx)

    article_b = to_tensor(article_b)
    extend_article_b = to_tensor(extend_article_b)
    title_b = to_tensor(title_b)

    return [article_b, extend_article_b, title_b, extend_vocab_b, extend_count_b]

def index_and_extend(source, target, source_vocab, target_vocab):

    extend_source = deepcopy(source)
    extend_vocab = [{} for _ in range(len(source))]
    extend_count = [0 for _ in range(len(source))]
    for sid, sentence in enumerate(source):
        for wid, word in enumerate(sentence):
            if word in source_vocab:
                source[sid][wid] = source_vocab[word]
            else:
                #raise KeyError('{} not in vocab'.format(word))
                source[sid][wid] = source_vocab['UNK']
            # make extend
            if word in target_vocab:
                extend_source[sid][wid] = target_vocab[word]
            elif word in extend_vocab[sid]:
                extend_source[sid][wid] = extend_vocab[sid][word]
            else:
                extend_vocab[sid][word] = len(target_vocab) + extend_count[sid]
                extend_source[sid][wid] = extend_vocab[sid][word]
                extend_count[sid] += 1

    for sid, sentence in enumerate(target):
        for wid, word in enumerate(sentence):
            if word in target_vocab:
                target[sid][wid] = target_vocab[word]
            elif word in extend_vocab[sid]:
                target[sid][wid] = extend_vocab[sid][word]
            else:
                target[sid][wid] = target_vocab['UNK']
                #raise KeyError('{} not in vocab'.format(word))

    return [source, extend_source, target, extend_vocab, extend_count]

def sampling(source, target, sl, tl, val_size):

    val_source, val_target, val_sl, val_tl = [], [], [], []
    for _ in range(val_size):
        idx = np.random.randint(0, len(source))
        val_source.append(source.pop(idx))
        val_target.append(target.pop(idx))
        val_sl.append(sl.pop(idx))
        val_tl.append(tl.pop(idx))

    return [val_source, val_target, val_sl, val_tl]


def padding(source, target, truncate=1000000):


    source_lengths = [min(len(x),truncate) for x in source]
    target_lengths = [min(len(x),truncate) for x in target]
    max_source = max(source_lengths)
    max_target = max(target_lengths)

    for i, _ in enumerate(source):
        source[i] += ['PAD' for e in range(max_source - source_lengths[i])]
        target[i] += ['PAD' for e in range(max_target - target_lengths[i])]
        source[i] = source[i][:max_source]
        target[i] = target[i][:max_target]

    return [source, target, source_lengths, target_lengths]

def sort_by_length(source, target):

    st = sorted(zip(source, target), key=lambda x:len(x[0]), reverse=True)
    source_s, target_s = zip(*st)
    if not isinstance(source_s, list):
        source_s = list(source_s)
    if not isinstance(target_s, list):
        target_s = list(target_s)

    return [source_s, target_s]

def sort_target(source, target):

    st = sorted(zip(source, target), key=lambda x:len(x[1]))
    source_s, target_s = zip(*st)
    if not isinstance(source_s, list):
        source_s = list(source_s)
    if not isinstance(target_s, list):
        target_s = list(target_s)

    return [source_s, target_s]

def sort_batch(source, target, source_lengths):

    st = sorted(zip(source, target, source_lengths), key=lambda x:x[2], reverse=True)
    source_s, target_s, source_lengths = zip(*st)
    if not isinstance(source_s, list):
        source_s = list(source_s)
    if not isinstance(target_s, list):
        target_s = list(target_s)
    if not isinstance(source_lengths, list):
        source_lengths = list(source_lengths)

    return [source_s, target_s, source_lengths]

def save_data(data, filename):
    with open(filename,'wb') as f:
        pickle.dump(data,f)

def load_data(filename):

    if os.path.exists(filename):
        with open(filename,'rb') as f:
            data = pickle.load(f)
        return data


def to_tensor(data):
    data1 = Variable(torch.LongTensor(data))
    return data1
