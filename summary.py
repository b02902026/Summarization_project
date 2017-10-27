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


def to_tensor(data):
    data1 = Variable(torch.LongTensor(data))
    return data1

def prepare_training(args):
    if args.CNN:
        article, title, word2idx, target2idx, maxl = read_cnn_dailymail(thres=args.train_size)
        tr_a = int(maxl[0] * 0.7)
    else:
        article, title, word2idx, target2idx, maxl = read_textfile(thres=args.train_size)
        tr_a = 1000000

    vocab_size = len(word2idx)
    article, title = utils.sort_by_length(article, title)
    article, title, source_lengths, target_lengths = utils.padding(article, title, tr_a)


    return [article, title, source_lengths, target_lengths, word2idx, target2idx]


def train(article, title, word2idx, target2idx, source_lengths, target_lengths,
          args, article_extend=None, extend_vocab=None, extend_counts=None):



    size_of_val = int(len(article)*0.05)

    val_article, val_title, val_source_lengths, val_target_lengths = \
        utils.sampling(article, title, source_lengths, target_lengths, size_of_val)



    batch_size = args.batch
    train_size = len(article)
    val_size = len(val_article)
    max_a = max(source_lengths)
    max_t = max(target_lengths)
    print("source vocab size:", len(word2idx))
    print("target vocab size:", len(target2idx))
    print("max a:{}, max t:{}".format(max_a, max_t))
    print("train_size:",train_size)
    print("val size:", val_size)
    print("batch_size:",batch_size)
    print("-"*30)
    use_coverage = False


    encoder = Encoder(len(word2idx))
    decoder = Decoder(len(target2idx),50)
    if os.path.exists("encoder_model"):
        print("Model existed. Loaded...")
        encoder_w = torch.load("encoder_model")
        encoder.load_state_dict(encoder_w)
        decoder_w = torch.load("decoder_model")
        decoder.load_state_dict(decoder_w)


    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    n_epoch = 5
    print("Making word index and extend vocab")
    #article, article_tar, title, ext_vocab_all, ext_count = indexing_word(article, title, word2idx, target2idx)
    #article = to_tensor(article)
    #article_extend = to_tensor(article_extend)
    #title = to_tensor(title)
    print("preprocess done")


    if args.use_cuda:
        encoder.cuda()
        decoder.cuda()

    print("start training")
    for epoch in range(n_epoch):
        total_loss = 0
        batch_n = int(train_size / batch_size)
        if epoch > 0:
            use_coverage = True
        for b in range(batch_n):
            # initialization
            batch_x = article[b*batch_size: (b+1)*batch_size]
            batch_y = title[b*batch_size: (b+1)*batch_size]
            #batch_x_ext = article_extend[b*batch_size: (b+1)*batch_size]
            batch_x, batch_x_ext, batch_y, extend_vocab, extend_lengths = \
                utils.batch_index(batch_x, batch_y, word2idx, target2idx)

            if args.use_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_x_ext = batch_x_ext.cuda()
            x_lengths = source_lengths[b*batch_size: (b+1)*batch_size]
            y_lengths = target_lengths[b*batch_size: (b+1)*batch_size]
            #ext_lengths = extend_counts[b*batch_size: (b+1)*batch_size]

            # work around to deal with length
            pack = pack_padded_sequence(batch_x_ext, x_lengths, batch_first=True)
            batch_x_ext_var, _ = pad_packed_sequence(pack, batch_first=True)
            current_loss = train_on_batch(encoder, decoder, optimizer,
                                         batch_x, batch_y, x_lengths, y_lengths,
                                         word2idx, target2idx, batch_x_ext_var,
                                          extend_lengths, use_coverage)

            batch_x = batch_x.cpu()
            batch_y = batch_y.cpu()
            batch_x_ext = batch_x_ext.cpu()

            print('epoch:{}/{}, batch:{}/{}, loss:{}'.format(epoch+1, n_epoch, b+1, batch_n, current_loss))
            if (b+1) % args.show_decode == 0:
                torch.save(encoder.state_dict(), 'encoder_model')
                torch.save(decoder.state_dict(), 'decoder_model')
                batch_x_val, batch_x_ext_val, batch_y_val, extend_vocab, extend_lengths = \
                    utils.batch_index(val_article, val_title, word2idx, target2idx)
                for i in range(5):
                    idx = np.random.randint(0,val_size)
                    decode.beam_search(encoder, decoder, batch_x_val[idx].unsqueeze(0),
                                batch_y_val[idx].unsqueeze(0),  word2idx, target2idx,
                                       batch_x_ext_val[idx], extend_lengths[idx],
                                       extend_vocab[idx])

                batch_x_val = batch_x_val.cpu()
                batch_y_val = batch_y_val.cpu()
                batch_x_ext_val = batch_x_ext_val.cpu()

            total_loss += current_loss
            print('-'*30)

    print()
    print("training finished")


def reverse_mapping(word2idx):

    idx2word = {v:k for (k,v) in word2idx.items()}

    return idx2word


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda",action="store_true")
    parser.add_argument("-beam_size",action="store", type=int)
    parser.add_argument("-show_decode",action="store", type=int)
    parser.add_argument("-batch",action="store", type=int)
    parser.add_argument("-train_size", action="store", type=int)
    parser.add_argument("--CNN",action="store_true")
    parser.add_argument("--giga",action="store_true")
    args = parser.parse_args()

    if os.path.exists('data/train_article.pkl'):
        article = utils.load_data('data/train_article.pkl')
        title = utils.load_data('data/train_title.pkl')
        s_lengths = utils.load_data('data/train_sl.pkl')
        t_lengths = utils.load_data('data/train_tl.pkl')
        word2idx = utils.load_data('data/vocab.pkl')
        target2idx = word2idx
    else:
        article, title, s_lengths, t_lengths, word2idx, target2idx = prepare_training(args)

    train(article, title, word2idx, target2idx, s_lengths, t_lengths, args)


