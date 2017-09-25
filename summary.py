from data_preprocess import *
from copy import deepcopy
from seq2seq import Encoder, Decoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import argparse

from greedy_decode import greedy_eval
import prepare_data as utils
import decode
from training import train_on_batch

def to_tensor(data):
    data1 = Variable(torch.LongTensor(data))
    return data1

def prepare_training():
    article, title, word2idx, target2idx, maxl = read_textfile()
    vocab_size = len(word2idx)
    article, title = utils.sort_by_length(article, title)
    #article, title = utils.sort_by_target(article, title)
    article, title, source_lengths, target_lengths = utils.padding(article, title)
    article, title = utils.indexing(article, title, word2idx, target2idx)

    return [article, title, source_lengths, target_lengths, word2idx, target2idx]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(article, title, word2idx, target2idx, source_lengths, target_lengths, args):

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
    decoder = Decoder(len(target2idx))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    n_epoch = 5
    print("Making word index and extend vocab")
    #article, article_tar, title, ext_vocab_all, ext_count = indexing_word(article, title, word2idx, target2idx)
    article = to_tensor(article)
    title = to_tensor(title)
    print("preprocess done")


    if args.USE_CUDA:
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
            if args.USE_CUDA:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            x_lengths = source_lengths[b*batch_size: (b+1)*batch_size]
            y_lengths = target_lengths[b*batch_size: (b+1)*batch_size]
            current_loss = train_on_batch(encoder, decoder, optimizer,
                                         batch_x, batch_y, x_lengths, y_lengths,
                                         word2idx, target2idx)

            print('epoch:{}/{}, batch:{}/{}, loss:{}'.format(epoch+1, n_epoch, b+1, batch_n, current_loss))
            if (b+1) % args.show_res == 0:
                torch.save(encoder.state_dict(), 'encoder.pth.tar')
                torch.save(decoder.state_dict(), 'decoder.pth.tar')
                for i in range(5):
                    decode.beam_search(encoder, decoder, article[i].unsqueeze(0),
                                title[i].unsqueeze(0),  word2idx, target2idx)
            total_loss += current_loss
            print('-'*30)

    print()
    print("training finished")
    for i in range(10):
        decode.greedy(encoder, decoder, article[i].unsqueeze(0),
                    title[i].unsqueeze(0), word2idx, target2idx)


def beam_search(input_seq, target, word2idx, target2idx,encoder,decoder,args):

    target2word = reverse_mapping(target2idx)
    decoded = []
    top_k_list = [] # will contain at most beam size ^ 2 instance
    top_k_pair = [] # will store (prev_word_id, value, pred_word, prev_hidden)
    trace_back = []
    encoder.train(False)
    decoder.train(False)
    max_length = input_seq.size(1)
    enc_out, enc_hidden = encoder(input_seq, None, [max_length])
    dec_input = Variable(torch.LongTensor([[word2idx['SOS']]]))
    if args.USE_CUDA:
        dec_input = dec_input.cuda()
    top_k_list.append(dec_input)
    dec_hidden = enc_hidden[0].unsqueeze(0)
    for i in range(max_length):
        for k,cand in enumerate(top_k_list):
            dec_output, dec_hidden = decoder(cand, dec_hidden, enc_out, max_length)
            v, idx = dec_output.cpu().data.topk(args.beam_size)
            for n in range(args.beam_size):
                top_k_pair.append((k, v[0][n], idx[0][n]))
        # select beam size number of instance
        top_k_pair = sorted(top_k_pair, key=lambda x:x[1], reverse=True)
        trace_back.append([(ins[0], ins[1],ins[2]) for ins in top_k_pair[:args.beam_size]])
        if top_k_pair[0][2] == word2idx['EOS'] or top_k_pair[1][2] == word2idx['EOS']:
            break
        top_k_pair, top_k_list = [], []
        for ins in trace_back[-1]:
            top_k_list.append(Variable(torch.LongTensor([[ins[2]]])).cuda())

        #dec_input = Variable(torch.LongTensor(idx))
        #if idx[0][0] == target2idx['EOS']:
        #    break
        #decoded.append(idx[0][0])
    #print(trace_back)
    decoded_words = beam_decode(target2word, trace_back)
    arti = ''
    idx2word = reverse_mapping(word2idx)
    for w in input_seq.data[0]:
        if idx2word[w] == 'EOS':    break
        arti += idx2word[w] +' '

    tar = ''
    for w in target.data[0]:
        if target2word[w] == 'EOS': break
        tar += target2word[w]+' '
    #print("-"*30)
    print("Original Paragraph:")
    print(arti)
    print("-"*30)
    print("target summary:")
    print(tar)
    print("-"*30)
    print("decoded:")
    print(decoded_words)
    print("-"*30)
    encoder.train(True)
    decoder.train(True)

def beam_decode(target2word, trace_back):
    decoded = []
    #decoded.append(trace_back[-1][0][2])
    prev = trace_back[-1][0][0]
    for i, lis in enumerate(reversed(trace_back[:-1])):
        decoded.append(lis[prev][2])
        prev = lis[prev][0]

    decoded_words = "".join([target2word[w]+' ' for w in reversed(decoded)])
    return decoded_words

def reverse_mapping(word2idx):

    idx2word = {v:k for (k,v) in word2idx.items()}

    return idx2word


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--USE_CUDA",action="store_true")
    parser.add_argument("-beam_size",action="store", type=int)
    parser.add_argument("-show_res",action="store", type=int)
    parser.add_argument("-batch",action="store", type=int)
    args = parser.parse_args()

    article, title, s_lengths, t_lengths, word2idx, target2idx = prepare_training()
    train(article, title, word2idx, target2idx, s_lengths, t_lengths, args)
