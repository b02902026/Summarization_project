from data_preprocess import *
from copy import deepcopy
from seq2seq import Encoder, Decoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import argparse
import os

from greedy_decode import greedy_eval
import prepare_data as utils
import decode
from training import train_on_batch

os.environ["CUDA_VISIBLE_DEVICE"] = "1"

def to_tensor(data):
    data1 = Variable(torch.LongTensor(data))
    return data1

def prepare_training(args):
    article, title, word2idx, target2idx, maxl = read_textfile(thres=args.train_size)
    vocab_size = len(word2idx)
    article, title = utils.sort_by_length(article, title)
    article, title, source_lengths, target_lengths = utils.padding(article, title)
    if args.pointer:
        article, extend_article, title, extend_vocab, extend_count = \
            utils.index_and_extend(article, title, word2idx, target2idx)
        return [article, extend_article, title, extend_vocab, extend_count,
                source_lengths, target_lengths, word2idx, target2idx]

    else:
        article, title = utils.indexing(article, title, word2idx, target2idx)
        return [article, title, source_lengths, target_lengths, word2idx, target2idx]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(article, title, word2idx, target2idx, source_lengths, target_lengths,
          args, article_extend=None, extend_vocab=None, extend_counts=None):

    batch_size = args.batch
    train_size = len(article)
    max_a = max(source_lengths)
    max_t = max(target_lengths)
    print("source vocab size:", len(word2idx))
    print("target vocab size:", len(target2idx))
    print("max a:{}, max t:{}".format(max_a, max_t))
    print("train_size:",train_size)
    print("batch_size:",batch_size)
    print("-"*30)

    encoder = Encoder(len(word2idx))
    decoder = Decoder(len(target2idx),max(extend_counts))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    n_epoch = 5
    print("Making word index and extend vocab")
    #article, article_tar, title, ext_vocab_all, ext_count = indexing_word(article, title, word2idx, target2idx)
    article = to_tensor(article)
    article_extend = to_tensor(article_extend)
    title = to_tensor(title)
    print("preprocess done")


    if args.use_cuda:
        encoder.cuda()
        decoder.cuda()

    print("start training")
    for epoch in range(n_epoch):
        total_loss = 0
        batch_n = int(train_size / batch_size)
        for b in range(batch_n):
            # initialization
            batch_x = article[b*batch_size: (b+1)*batch_size]
            batch_y = title[b*batch_size: (b+1)*batch_size]
            batch_x_ext = article_extend[b*batch_size: (b+1)*batch_size]
            if args.use_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_x_ext = batch_x_ext.cuda()
            x_lengths = source_lengths[b*batch_size: (b+1)*batch_size]
            y_lengths = target_lengths[b*batch_size: (b+1)*batch_size]
            ext_lengths = extend_counts[b*batch_size: (b+1)*batch_size]
            # work around to deal with length
            pack = pack_padded_sequence(batch_x_ext, x_lengths, batch_first=True)
            batch_x_ext, _ = pad_packed_sequence(pack, batch_first=True)
            current_loss = train_on_batch(encoder, decoder, optimizer,
                                         batch_x, batch_y, x_lengths, y_lengths,
                                         word2idx, target2idx, batch_x_ext, ext_lengths)

            print('epoch:{}/{}, batch:{}/{}, loss:{}'.format(epoch+1, n_epoch, b+1, batch_n, current_loss))
            if (b+1) % args.show_decode == 0:
                torch.save(encoder.state_dict(), 'encoder.pth.tar')
                torch.save(decoder.state_dict(), 'decoder.pth.tar')
                for i in range(5):
                    decode.beam_search(encoder, decoder, article[i].unsqueeze(0),
                                title[i].unsqueeze(0),  word2idx, target2idx,
                                       article_extend[i], extend_counts[i],
                                       extend_vocab[i])
            total_loss += current_loss
            print('-'*30)

    print()
    print("training finished")
    for i in range(10):
        decode.greedy(encoder, decoder, article[i].unsqueeze(0),
                    title[i].unsqueeze(0), word2idx, target2idx)



def reverse_mapping(word2idx):

    idx2word = {v:k for (k,v) in word2idx.items()}

    return idx2word


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda",action="store_true")
    parser.add_argument("--pointer",action="store_true")
    parser.add_argument("-beam_size",action="store", type=int)
    parser.add_argument("-show_decode",action="store", type=int)
    parser.add_argument("-batch",action="store", type=int)
    parser.add_argument("-train_size", action="store", type=int)
    args = parser.parse_args()
    if args.pointer:
        article, article_extend, title, extend_vocab, extend_c,s_lengths, t_lengths,\
            word2idx, target2idx = prepare_training(args)
        train(article, title, word2idx, target2idx, s_lengths, t_lengths,args,
              article_extend, extend_vocab, extend_c)
    else:
        article, title, s_lengths, t_lengths, word2idx, target2idx = prepare_training(args)
        train(article, title, word2idx, target2idx, s_lengths, t_lengths, args)




