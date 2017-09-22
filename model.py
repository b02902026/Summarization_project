import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_dim = 256
        self.embedding_dim = 128
        self.embedder = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, word, hidden, lengths):
        emb = self.embedder(word)
        packed = pack_padded_sequence(emb, lengths,batch_first=True)
        output, h = self.gru(packed, None)
        outputs, output_lengths = pad_packed_sequence(output, batch_first=True)
        return outputs, h

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_dim))

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_a):
        super(Decoder, self).__init__()
        self.hidden_dim = 256
        self.embedding_dim = 128
        self.embedder = nn.Embedding(vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, 1, batch_first=True)
        self.attention_linear = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.v = nn.Parameter(torch.randn(self.hidden_dim))
        self.concat_linear = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, vocab_size)
        self.enc_seq_len = max_a
        self.W_context = nn.Parameter(torch.randn(self.hidden_dim*2, 1))
        self.W_dec = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.W_input = nn.Parameter(torch.randn(self.embedding_dim, 1))
        self.b_gen = nn.Parameter(torch.randn(1))
        self.vocab_size = vocab_size

    def forward(self, word, hidden, enc_output, max_length, max_ext_lengths, ext_word):

        batch_size = word.size(0)
        embedded = self.embedder(word)
        output, h = self.gru(embedded, hidden)
        # TESTING vanilla seq2seq
        return F.softmax(self.output_layer(output.squeeze(1))), h

        attention_energy = Variable(torch.zeros(batch_size, max_length)).cuda()
        for t in range(max_length):
            concated = torch.cat((enc_output[:,t,:],output.squeeze(1)),1)
            project = F.tanh(self.attention_linear(concated))
            linear_transform = torch.mv(project, self.v)
            attention_energy[:,t] = linear_transform

        self.attention_matrix = F.softmax(attention_energy).unsqueeze(1)
        # [B,1,seq] * [B,seq,2H] = [B,1,2H] -> [B,2H]
        context = self.attention_matrix.bmm(enc_output).squeeze(1)
        concat = torch.cat((output.squeeze(1), context),1)
        concat_out = self.concat_linear(concat)
        out = self.output_layer(concat_out)
        output_word = F.softmax(out)    # the distribution of source vocab
        return output_word, h

        p_gen = context.mm(self.W_context) + output.squeeze(1).mm(self.W_dec) + embedded.squeeze(1).mm(self.W_input) # shape [B]
        p_gen = torch.add(p_gen, self.b_gen)    # add bias
        p_gen = F.sigmoid(p_gen)
        p_gen_neg = p_gen.expand(batch_size, max_length)   # shape [B, seq]
        p_gen = p_gen.expand(batch_size, self.vocab_size)  # shape [B, vocab]

        word_distribution = p_gen * output_word # shape [B,vocab]
        p_point = torch.neg(torch.add(p_gen_neg,-1))
        atten_distribution = p_point * self.attention_matrix.squeeze(1) # shape [B, seq_len]
        print("gen",p_gen.data[0,0],"point", p_point.data[0,0])

        extend_matrix = Variable(torch.zeros(batch_size, max_ext_lengths)).cuda()
        extend_vocab = torch.cat((word_distribution, extend_matrix),1)  # shape [B, vocab size + extend oov size]

        # ext_word is the source article indexed by target vocab and ext vocab
        # the shape should looks like [B, seq_len]
        indice = ext_word
        attn_scatter_zeros = Variable(torch.zeros(batch_size, self.vocab_size + max_ext_lengths)).cuda()
        attn_scatter = attn_scatter_zeros.scatter_(1, indice, atten_distribution).cuda()   # shape [B, ext]
        print("attention", atten_distribution.data[0])
        print("indice", indice.data[0])
        print("attention scatter", attn_scatter.data[0].topk(5))
        print("word", word_distribution.data[0].topk(5))
        output_word = torch.add(attn_scatter, extend_vocab)
        print("final output",output_word.data[0].topk(5))

        return output_word, h

