import torch
import torch.nn as nn
from torch.autograd import Variable

def train_on_batch(encoder, decoder, opt, source, target, source_lengths,
                   target_lengths, source_vocab, target_vocab, source_extend=None,
                   extend_lengths=None, use_coverage=False):

    # Argument : source should be (batch, seq) ; target should be (batch, seq)
    batch_size = len(source)
    #loss_fn = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
    loss_fn = nn.NLLLoss(size_average=False, ignore_index=0)
    loss = 0
    opt.zero_grad()

    encoder_output, encoder_hidden, encoder_cell = encoder(source, source_lengths)
    decoder_hidden = encoder_hidden[0]
    decoder_cell = encoder_cell[0]
    decoder_input = Variable(torch.LongTensor([target_vocab['SOS']]*batch_size)).view(batch_size,1).cuda()
    coverage_vector = Variable(torch.zeros(batch_size, max(source_lengths))).cuda()
    coverage_loss = 0

    for step in range(max(target_lengths)):
        word_dist, decoder_hidden, decoder_cell, attn = decoder(decoder_input, decoder_hidden,
                                                          decoder_cell, encoder_output,
                                                          max(source_lengths),
                                                          source_extend,
                                                          max(extend_lengths))

        loss += loss_fn(word_dist, target[:,step])
        min_cover = torch.min(torch.stack([coverage_vector, attn]),0)[0]
        coverage_loss += torch.sum(min_cover)
        coverage_vector += attn

        decoder_input = target[:,step].cuda()

    if use_coverage:
        loss += coverage_loss
    loss.backward()
    opt.step()


    return loss.data[0] / sum(target_lengths)





