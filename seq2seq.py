import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = 256
        self.embedding_size = 128
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, 1, batch_first=True, bidirectional=True)

    def forward(self, word, lengths):

        embedded = self.embed(word) # shape (batch, seq, embedding_size)
        inputs = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        zero_state = Variable(torch.zeros(2,embedded.size(0),self.hidden_size)).cuda()
        outputs, states = self.rnn(inputs, (zero_state, zero_state))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, states[0], states[1]

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = 256
        self.embedding_size = 128
        self.target_vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.rnn_cell = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.attn_linear = nn.Linear(self.hidden_size*3,self.hidden_size)
        self.v = nn.Parameter(torch.randn([self.hidden_size]))
        self.project = nn.Linear(self.hidden_size*3,self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, word, hidden, cell, encoder_output, max_source_len):

        batch_size = word.size(0)
        embedded = self.embed(word) # shape (batch, 1 ,embeding_size)
        embedded = embedded.squeeze(1)
        h_t, c_t = self.rnn_cell(embedded, (hidden, cell))
        h_t = F.tanh(h_t)
        #final_out = self.output_projection(out)
        #word_distribution = final_out
        attention_matrix = Variable(torch.zeros(batch_size,max_source_len)).cuda()  # shape (batch, seq_len)
        for step in range(max_source_len):
            concated = torch.cat((encoder_output[:,step,:],h_t),1)  # shape (batch, hidden_size * 3)
            concated = F.tanh(self.attn_linear(concated))   # shape (batch, hidden_size)
            e = concated.mv(self.v)
            attention_matrix[:,step] = e
        # attention matrix is (batch, seq_len), so unsqueeze it to be (batch, 1,
        # seq_len) to perform batch mm with encoder hidden state (batch,
        # seq_len, hidden) -> context will be (batch, 1, hidden) then squeeze
        context = F.softmax(attention_matrix).unsqueeze(1).bmm(encoder_output).squeeze(1)
        cat = torch.cat((context, h_t),1)
        cat_project = self.project(cat)
        word_distribution = self.output_projection(cat_project)

        return word_distribution, h_t, c_t


