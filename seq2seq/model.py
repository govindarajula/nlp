import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, emb_dim, hidden_size, decoder_hidden_size, vocab_size):
        """
        Args:
            emb_dim (int): Embedding size
            hidden_size (int): Encoder hidden size
            decoder_hidden_size (int): Decoder hidden size
            vocab_size (int): Size of vocab
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size, bidirectional=True, dropout=0.2)

        self.fc = nn.Linear(hidden_size * 2, decoder_hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x(tensor): Input sentence of size (src len, batch size)
        Returns: Encoder output (src len, batch size, hidden * 2) and hidden (batch size, decoder_hidden_size)
        """
        x = self.dropout(self.embedding(x.long()))                                 # (src len, batch size, emb_dim)
        x, h = self.gru(x)
        h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)                    # concat last forward and backward cell's outputs
        h = torch.tanh(self.fc(h))
        return x, h


class Decoder(nn.Module):

    def __init__(self, emb_dim, vocab_size, hidden_size, encoder_hidden_size):
        """
        Args:
            emb_dim (int): Embedding size
            vocab_size (int): Size of vocab
            hidden_size (int): Decoder hidden size
            encoder_hidden_size (int): Encoder hidden size
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU((encoder_hidden_size * 2) + emb_dim, hidden_size, dropout=0.2)

        self.attn_fc = nn.Linear((encoder_hidden_size * 2) + hidden_size, 1)

        self.fc = nn.Linear((encoder_hidden_size * 2) + hidden_size + emb_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h, encoder_outputs):
        """
        Args:
            x: (batch_size, 1)
            h: (batch_size, hidden_size)
            encoder_outputs: (src len, batch size, enc_hid_dim * 2)
        Returns: Decoder output (batch size, vocab) and hidden (1, batch size, hidden)
        """
        # Attention
        src_len = encoder_outputs.shape[0]
        h_rep = h.unsqueeze(1).repeat(1, src_len, 1)                    # (batch size, src len, dec hid dim) 5, 100, 20
        encoder_outputs = encoder_outputs.permute(1, 0, 2)              # (batch size, src len, enc_hid_dim * 2) 5, 100, 100
        weights = torch.tanh(self.attn_fc(torch.cat((h_rep, encoder_outputs), dim=2)))     # (batch size, src len, 1)
        embedded = self.embedding(x.long()).squeeze(1)                         # (batch_size, emb_dim)
        embedded = self.dropout(embedded)

        weighted = torch.bmm(encoder_outputs.permute(0, 2, 1), weights).squeeze(2)       # (batch size, enc_hid_dim * 2)
        x = torch.cat((embedded, weighted), dim=1)                      # (batch size, emb_dim + enc_hid_dim*2)
        x, h = self.gru(x.unsqueeze(0), h.unsqueeze(0))                 # src size = 1

        assert (x == h).all()

        x = self.fc(torch.cat((x.squeeze(0), weighted, embedded), dim=1))   # (batch size, vocab)

        return x, h.squeeze(0)


if __name__ == "__main__":
    encoder = Encoder(3, 50, 40, 1000)
    inp = torch.randint(0, 1000, (100, 5))
    output = encoder(inp)
    print(output[0].size(), output[1].size())

    decoder = Decoder(3, 1000, 40, 50)
    print(decoder(torch.randint(0, 1000, (5, 1)), output[1], output[0]))
