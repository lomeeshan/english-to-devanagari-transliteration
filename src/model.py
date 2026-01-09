import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    """
    Character-level LSTM encoder.
    Input:  (batch, src_len) token IDs
    Output: final hidden + cell states for the decoder
    """
    def __init__(self, src_vocab_size: int, emb_dim: int, hidden_dim: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src: (B, S)
        emb = self.embedding(src)         # (B, S, E)
        _, (h, c) = self.lstm(emb)        # h,c: (1, B, H)
        return h, c


class Decoder(nn.Module):
    """
    Character-level LSTM decoder.
    Takes previous target token + encoder state and predicts next token.
    """
    def __init__(self, tgt_vocab_size: int, emb_dim: int, hidden_dim: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, tgt_in: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tgt_in: (B, 1) previous token IDs (one time step)
        h, c : (1, B, H)
        returns:
          logits: (B, 1, V)
          h, c: updated states
        """
        emb = self.embedding(tgt_in)        # (B, 1, E)
        out, (h, c) = self.lstm(emb, (h, c))# out: (B, 1, H)
        logits = self.fc_out(out)          # (B, 1, V)
        return logits, h, c


class Seq2Seq(nn.Module):
    """
    Minimal Seq2Seq:
    - Encoder produces final hidden/cell
    - Decoder predicts tokens step-by-step using teacher forcing during training
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, sos_id: int, eos_id: int, pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """
        src: (B, S)
        tgt: (B, T) includes <SOS> ... <EOS>
        returns logits for each step (B, T-1, V) predicting tgt[1:] from tgt[:-1]
        """
        B, T = tgt.shape
        device = tgt.device
        V = self.decoder.fc_out.out_features

        h, c = self.encoder(src)

        # predict 1..T-1 (next tokens), given inputs 0..T-2
        logits_all = torch.zeros(B, T - 1, V, device=device)

        # First decoder input token is <SOS> (which is tgt[:,0])
        dec_input = tgt[:, 0].unsqueeze(1)  # (B, 1)

        for t in range(1, T):
            logits, h, c = self.decoder(dec_input, h, c)   # logits: (B,1,V)
            logits_all[:, t - 1, :] = logits.squeeze(1)

            # Decide next input: teacher forcing or model prediction
            use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = logits.argmax(dim=-1)  # (B,1)

            dec_input = tgt[:, t].unsqueeze(1) if use_teacher else top1

        return logits_all

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 30) -> torch.Tensor:
        """
        Greedy decoding for inference.
        src: (B, S)
        returns predicted token IDs including SOS and EOS: (B, <=max_len)
        """
        device = src.device
        h, c = self.encoder(src)

        B = src.size(0)
        dec_input = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
        outputs = [dec_input]

        for _ in range(max_len - 1):
            logits, h, c = self.decoder(dec_input, h, c)
            next_token = logits.argmax(dim=-1)  # (B,1)
            outputs.append(next_token)
            dec_input = next_token

        return torch.cat(outputs, dim=1)
