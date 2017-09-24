from copy import deepcopy
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

def index_and_extend(source, target, source_vocab, target_vocab):

    extend_source = deepcopy(source)
    extend_vocab = [{} for _ in range(len(source))]
    extend_count = [0 for _ in range(len(source))]
    for sid, sentence in enumerate(source):
        for wid, word in enumerate(sentence):
            if word in source_vocab:
                source[sid][wid] = source_vocab[word]
            else:
                raise KeyError

            if word in target_vocab:
                extend_source[sid][wid] = target_vocab[word]
            elif word in extend_vocab[sid]:
                extend_source[sid][wid] = extend_source[sid][word]
            else:
                extend_vocab[word] = len(target_vocab) + extend_count[sid]
                extend_source[sid][wid] = extend_vocab[sid][word]
                extend_count[sid] += 1

    for sid, sentence in enumerate(target):
        for wid, word in enumerate(sentence):
            if word in target_vocab:
                target[sid][wid] = target_vocab[word]
            else:
                raise KeyError

    return [source, extend_source, target, extend_vocab, extend_count]

def padding(source, target):

    source_lengths = [len(x) for x in source]
    target_lengths = [len(x) for x in target]
    max_source = max(source_lengths)
    max_target = max(target_lengths)

    for i, _ in enumerate(source):
        source[i] += ['PAD' for e in range(max_source - source_lengths[i])]
        target[i] += ['PAD' for e in range(max_target - target_lengths[i])]

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


