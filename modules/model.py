import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int, 
                 dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(batch_first = True, 
                            num_layers = num_layers, 
                            dropout = self.dropout, 
                            hidden_size = self.hidden_dim, 
                            input_size = self.input_dim)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                lens: torch.Tensor):
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        out, hidden = self.lstm(x)
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.linear(out), lens

class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int, 
                 dropout: float, 
                 sos_token_id: int, 
                 eos_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.lstm = nn.LSTM(batch_first = True, 
                            num_layers = num_layers, 
                            dropout = self.dropout, 
                            hidden_size = self.hidden_dim, 
                            input_size = self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: torch.Tensor = None, 
                lens: torch.Tensor = None):
        
        x = self.embedding(x)
        if lens is None:
            out, hidden = self.lstm(x, hidden)
            return self.linear(out), hidden
        else:
            x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
            out, hidden = self.lstm(x, hidden)
            out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return out, hidden
        
class RNNT(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 input_dim: int,
                 hidden_dim: int, 
                 output_dim: int, 
                 num_enc_layers: int, 
                 num_dec_layers: int, 
                 dropout: float, 
                 sos_token_id: int, 
                 eos_token_id: int):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.encoder = Encoder(self.input_dim, 
                               self.hidden_dim, 
                               self.output_dim, 
                               self.num_enc_layers, 
                               self.dropout)
        self.decoder = Decoder(self.vocab_size, 
                               self.hidden_dim, 
                               self.output_dim, 
                               self.num_dec_layers, 
                               self.dropout, 
                               self.sos_token_id,
                               self.eos_token_id)
        self.classifier = nn.Linear(self.hidden_dim << 1, self.vocab_size)
    
    def forward(self, 
                enc_x: torch.Tensor,
                enc_lens: torch.Tensor, 
                dec_x: torch.Tensor, 
                dec_lens: torch.Tensor):
        enc_x, _ = self.encoder(enc_x, enc_lens)
        dec_x, _ = self.decoder(dec_x, dec_lens)
        return self.joint(enc_x, dec_x)
    
    def joint(self, 
              enc_x: torch.Tensor, 
              dec_x: torch.Tensor):
        if enc_x.dim() == 3 and dec_x.dim() == 3:
            enc_x = enc_x.unsqueeze(2).repeat([1, 1, dec_x.size(1), 1])
            dec_x = dec_x.unsqueeze(1).repeat([1, enc_x.size(1), 1, 1])
        return self.classifier(torch.cat((enc_x, dec_x), dim=-1))
    
    def decode(self, 
               enc_x: torch.Tensor, 
               seq_len: int):
        preds, hidden = [], None
        dec_x = enc_x.new_tensor([[self.sos_token_id]], dtype=torch.long)
        for i in range(seq_len):
            dec_out, hidden = self.decoder(dec_x, hidden=hidden)
            logits = self.joint(enc_x[i].view(-1), dec_out.view(-1)).softmax(dim=-1)
            pred = logits.argmax(dim=-1).item()
            preds.append(pred)
            dec_x = logits.new_tensor([[pred]], dtype=torch.long)
        return logits, torch.Tensor(preds)
        
    def evaluation(self,
             enc_x: torch.Tensor,
             enc_lens: torch.Tensor):
        out = []
        enc_x, enc_lens = self.encoder(enc_x, enc_lens)
        seq_len = enc_x.size(1)
        for sample in enc_x:
            logits, dec_x = self.decode(sample, seq_len)
            out.append(dec_x)

        out = torch.stack(out, dim=1)
        return out