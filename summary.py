from data_preprocess import *
from copy import deepcopy
from model import Encoder, Decoder
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import argparse

from greedy_decode import greedy_eval

def to_tensor(data):
    data1 = Variable(torch.LongTensor(data))
    return data1

def prepare_training():
    article, title, word2idx, target2idx, maxl = read_textfile()
    vocab_size = len(word2idx)
    return [article, title, word2idx, target2idx, maxl, vocab_size]

def sort_pad(source, target, word2idx, maxl):
    pairs = sorted(zip(source, target), key=lambda x: len(x[0]),reverse=True)
    article, title = zip(*pairs)
    article, title, source_lengths = padding(list(article), list(title), word2idx, maxl)
    return [article, title, source_lengths]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def training(article, title, word2idx, target2idx, maxl, vocab_size, source_lengths, args):

    batch_size_t = 32
    print("source vocab size:", vocab_size)
    print("target vocab size:", len(target2idx))
    max_a, max_t = maxl
    print("max a:{}, max t:{}".format(max_a, max_t))
    train_size = len(article)
    print("train_size:",train_size)
    print("batch_size:",batch_size_t)
    print("-"*30)

    encoder = Encoder(vocab_size)
    decoder = Decoder(len(target2idx), max_a)
    loss_fn = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=0.001)
    n_epoch = 50
    print("Making word index and extend vocab")
    article, article_tar, title, ext_vocab_all, ext_count = indexing_word(article, title, word2idx, target2idx)
    article = to_tensor(article)
    title = to_tensor(title)
    #article_tar = to_tensor(article_tar)
    print("preprocess done")
    tar2word = reverse_mapping(target2idx)

    if args.USE_CUDA:
        encoder.cuda()
        decoder.cuda()
    print("start training")
    for epoch in range(n_epoch):
        total_loss = 0
        batch_n = int(train_size / batch_size_t)
        for b in range(batch_n):
            # initialization
            loss = 0
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            # split the batch and move to GPU
            article_b = article[b*batch_size_t: (b+1)*batch_size_t].cuda()
            title_b = title[b*batch_size_t: (b+1)*batch_size_t].cuda()
            # split the kength vector and extend vocab
            ext_vocab = ext_vocab_all[b*batch_size_t: (b+1)*batch_size_t]
            ext_word_length = ext_count[b*batch_size_t: (b+1)*batch_size_t]
            s_lengths_b = source_lengths[b*batch_size_t: (b+1)*batch_size_t]
            batch_size = len(article_b)
            # truncate the padding to match the shape of packed sequence
            article_tar_b = article_tar[b*batch_size_t: (b+1)*batch_size_t]
            article_tar_b = [s[:max(s_lengths_b)] for s in article_tar_b]
            article_tar_b = to_tensor(article_tar_b).cuda()

            # start running the graph
            enc_hidden = encoder.init_hidden(batch_size).cuda()
            enc_out, enc_hidden = encoder(article_b, enc_hidden, s_lengths_b)
            print(enc_out.size())
            dec_hidden = enc_hidden[0].unsqueeze(0)
            dec_input = Variable(torch.LongTensor([[word2idx['SOS']] * batch_size])).view(batch_size,1)
            if args.USE_CUDA:
                dec_input = dec_input.cuda()

            for i in range(max_t):
                dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_out, max(s_lengths_b), max(ext_word_length), article_tar_b)
                dec_input = title_b[:,i].unsqueeze(1)
                loss += loss_fn(dec_output, title_b[:,i])
                if args.USE_CUDA:
                    dec_input = dec_input.cuda()

            loss.backward()
            optimizer_e.step()
            optimizer_d.step()
            current_loss = (loss.data[0] / sum(s_lengths_b) )
            total_loss += loss.data[0]

            print('\repoch:{}/{}, batch:{}/{}, loss:{}'.format(epoch+1, n_epoch, b+1, batch_n, current_loss), end='')
            if b % 100 == 0 and b != 0:
                torch.save(encoder.state_dict(), 'encoder.pth.tar')
                torch.save(decoder.state_dict(), 'decoder.pth.tar')
                print('\n'+'-'*30)
                for i in range(3):
                    greedy_eval(article[i].unsqueeze(0), title[i].unsqueeze(0), article_tar[i], word2idx, target2idx, encoder, decoder, ext_vocab_all[i], args)
        print("\naverage loss:" ,total_loss/sum(source_lengths))
        print('-'*30)
        for i in range(5):
            #pass
            greedy_eval(article[i].unsqueeze(0), title[i].unsqueeze(0), article_tar[i], word2idx, target2idx, encoder, decoder, ext_vocab_all[i], args)

    print("-"*30+"training finished"+'-'*30)
    for i in range(10):
        greedy_eval(article[i].unsqueeze(0), title[i].unsqueeze(0), word2idx, target2idx, encoder, decoder, ext_vocab_all[i], args)


def beam_search(input_seq, target, word2idx, target2idx,encoder,decoder,args):

    target2word = reverse_mapping(target2idx)
    decoded = []
    top_k_list = [] # will contain at most beam size ^ 2 instance
    top_k_pair = [] # will store (prev_word_id, value, pred_word)
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
    args = parser.parse_args()

    article, title, word2idx, target2idx, maxl, vocab_size = prepare_training()
    article, title, lengths = sort_pad(article, title, word2idx,  maxl)
    training(article, title, word2idx, target2idx, maxl, vocab_size, lengths, args)
