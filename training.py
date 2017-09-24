import torch
import torch.nn as nn
from torch.autograd import Variable

def train_on_batch(encoder, decoder, opt, source, target, source_lengths,
                   target_lengths, source_vocab, target_vocab):

    # Argument : source should be (batch, seq) ; target should be (batch, seq)
    batch_size = len(source)
    loss_fn = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
    loss = 0
    opt.zero_grad()

    encoder_output, encoder_hidden, encoder_cell = encoder(source, source_lengths)
    decoder_hidden = encoder_hidden[0]
    decoder_cell = encoder_cell[0]
    decoder_input = Variable(torch.LongTensor([target_vocab['SOS']]*batch_size)).view(batch_size,1).cuda()

    for step in range(max(target_lengths)):
        word_dist, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden,
                                                          decoder_cell, encoder_output,
                                                          max(source_lengths))
        loss += loss_fn(word_dist, target[:,step])
        decoder_input = target[:,step].cuda()

    loss.backward()
    opt.step()

    return loss.data[0] / sum(target_lengths)





