import torch
from torch.autograd import Variable
def greedy_eval(input_seq, target, pointed, word2idx, target2idx,encoder,decoder,ext_vocab,args):


    target2word = reverse_mapping(target2idx)
    ext2word = reverse_mapping(ext_vocab)
    decoded = []
    encoder.train(False)
    decoder.train(False)


    max_length = input_seq.size(1)
    input_seq = input_seq.cuda()
    target = target.cuda()
    pointed = Variable(torch.LongTensor([pointed])).cuda()
    enc_out, enc_hidden = encoder(input_seq, None, [max_length])
    dec_input = Variable(torch.LongTensor([[word2idx['SOS']]]))
    if args.USE_CUDA:
        dec_input = dec_input.cuda()
    dec_hidden = enc_hidden[0].unsqueeze(0)
    for i in range(max_length):
        dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_out, max_length, len(ext_vocab), pointed)
        v, idx = dec_output.cpu().data.topk(1)
        dec_input = Variable(torch.LongTensor(idx)).cuda()
        if idx[0][0] == target2idx['EOS']:
            break
        decoded.append(idx[0][0])

    #print(decoded)
    decoded_words = "".join([target2word[e] + ' ' if e in target2word else ext2word[e] + ' ' for e in decoded])
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

def reverse_mapping(word2idx):

    idx2word = {v:k for (k,v) in word2idx.items()}

    return idx2word
