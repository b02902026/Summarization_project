import torch
from torch.autograd import Variable
from torch.nn import functional as F
from math import log
import numpy as np
from rouge import Rouge

def greedy(encoder, decoder, source, target, source_vocab, target_vocab,
           source_mapping=None, ext_length=None, ext_vocab=None):

    decoded = []
    seq_len = source.size(1)
    source = source.cuda()
    target = target.cuda()
    source_mapping = source_mapping.cuda().unsqueeze(0)
    output, h_n, c_n = encoder(source,[seq_len])
    decoder_h = h_n[0]
    decoder_c = c_n[0]
    decoder_input = Variable(torch.LongTensor([[target_vocab['SOS']]])).cuda()
    for step in range(seq_len):
        decode_output, decoder_h, decoder_c = decoder(decoder_input, decoder_h,
                                                      decoder_c, output,seq_len,
                                                      source_mapping, ext_length)
        v, idx = decode_output.cpu().data.topk(1)
        if idx[0][0] == target_vocab['EOS']:
            break
        decoded.append(idx[0][0])
        decoder_input = Variable(torch.LongTensor(idx)).cuda()

    # make inverse vocab
    reverse_src = reverse_mapping(source_vocab)
    reverse_tar = reverse_mapping(target_vocab)
    reverse_ext = reverse_mapping(ext_vocab)

    decoded_string = "".join([reverse_tar[w]+' ' if w in reverse_tar else reverse_ext[w]+' ' for w in decoded])
    print_result(source, target, decoded_string, reverse_src, reverse_tar,reverse_ext)

def beam_search(encoder, decoder, source, target, source_vocab, target_vocab,
                source_mapping=None, ext_length=None, ext_vocab=None):

    # candidate will be tuple (word, h_t, c_t), h_t and c_t are hidden and cell
    # state with the input word; and the output will be store in decoded pool in
    # the form of tuple (word, h_t_plus_1, c_t_plus_1, predict_word, prob, k)
    candidate = []
    decoded_pool = []
    vocab_prob = []
    trace_back = []
    beam_size = 4
    tail_pos = []
    ended = 0
    seq_len = source.size(1) if source.cpu().data.numpy()[0][-1] != 0 else int(np.where(source.data.cpu().numpy()==0)[1][0])
    source = source.cuda()
    target = target.cuda()
    source_mapping = source_mapping.cuda().unsqueeze(0)
    encoder_output, h_n, c_n = encoder(source,[seq_len])
    decoder_h = h_n[0]
    decoder_c = c_n[0]
    decoder_input = Variable(torch.LongTensor([[target_vocab['SOS']]])).cuda()
    candidate.append((decoder_input,decoder_h,decoder_c,0))
    for step in range(seq_len-1):
        for k, cand in enumerate(candidate):
            if cand == 'DUMMY': continue
            word, h_t, c_t, last_prob = cand[0], cand[1], cand[2], cand[3]
            decoder_out, decoder_h, decoder_c, attn = decoder(word, h_t, c_t,
                                                        encoder_output, seq_len,
                                                        source_mapping, ext_length)
            #decoder_out = F.log_softmax(decoder_out)

            summed_decoder_output = decoder_out.cpu().data + last_prob
            #vocab_prob.append(decoder_out.squeeze(0).numpy())
            prob, idx = summed_decoder_output.topk(beam_size)
            for next_word, p in zip(idx[0], prob[0]):
                decoded_pool.append((word, decoder_h, decoder_c,
                                     next_word, p, k))

        decoded_pool = sorted(decoded_pool, key=lambda x: x[4], reverse=True)
        decoded_pool = decoded_pool[:beam_size]
        candidate = []
        current_record = []
        for i, (word, h, c, next_word, p, k) in enumerate(decoded_pool):
            if next_word == source_vocab['EOS']:
                beam_size -= 1
                current_record.append((next_word, p, k))
                tail_pos.append((step, i))
                candidate.append('DUMMY')
            else:
                candidate.append((Variable(torch.LongTensor([[next_word]])).cuda(), h, c, p))
                current_record.append((next_word, p, k))
                if step == seq_len - 2:
                    tail_pos.append((step, i))

        decoded_pool = []
        trace_back.append(current_record[:])
        if beam_size == 0:
            break

    reverse_src = reverse_mapping(source_vocab)
    reverse_tar = reverse_mapping(target_vocab)
    reverse_ext = reverse_mapping(ext_vocab)
    decoded_string = beam_decode(trace_back, reverse_tar, reverse_ext, tail_pos, beam_size)
    print_result(source, target, decoded_string, reverse_src, reverse_tar, reverse_ext)


def beam_decode(trace_back, reverse_tar, reverse_ext,tail_pos, topk=16):

    decoded_all = ''
    probs = []
    decoded_store = []
    for k in range(4):

        prob = 0
        start_step = tail_pos[k][0]
        start_pos = tail_pos[k][1]
        decoded = []

        prev_word = trace_back[start_step][start_pos][2]
        cur_word = trace_back[start_step][start_pos][0]
        prob = trace_back[start_step][start_pos][1]

        if cur_word != 2:
            decoded.append(cur_word)
        for back_step in reversed(trace_back[:start_step]):
            decoded.append(back_step[prev_word][0])
            prev_word = back_step[prev_word][2]

        decoded = list(reversed(decoded))
        decoded_store.append(decoded[:])
        probs.append(prob / (len(decoded)+1))
        #decoded_all += "({}) ".format(k+1)
        #decoded_all += "".join([reverse_tar[w]+' ' if w in reverse_tar else reverse_ext[w]+' ' for w in decoded])
        #decoded_all += '\n' + '-'*30 + '\n'

    max_decoded_index = np.argmax(probs)
    decoded_all = "".join([reverse_tar[w]+' ' if w in reverse_tar else reverse_ext[w]+' ' for w in decoded_store[max_decoded_index]])

    return decoded_all

def get_score(decoded, target):
    rouge = Rouge()
    scores = rouge.get_scores(decoded, target)
    return scores

def print_result(source, target, decoded_string, reverse_src, reverse_tar, reverse_ext):

    source_string , target_string = [], []
    for w in source.data[0]:
        if w == 2:    break
        source_string.append(reverse_src[w])
    for w in target.data[0]:
        if w == 2:    break
        if w not in reverse_tar:
            target_string.append(reverse_ext[w])
        else:
            target_string.append(reverse_tar[w])

    source_string = "".join(w+' ' for w in source_string)
    target_string = "".join(w+' ' for w in target_string)
    score = get_score(decoded_string, target_string)
    # print the ground truth and decoded
    print('-'*30)
    print("source:")
    print(source_string)
    print('-'*30)
    print("ground truth:")
    print(target_string)
    print('-'*30)
    print("decoded:")
    print(decoded_string)
    print('-'*30)
    print("score:")
    print(score[0])


def reverse_mapping(voc):
    re = {v:k for k,v in voc.items()}
    return re
