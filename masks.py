
import torch
# sequnce length
def create_padding_mask(seq,pad_index=0):
    # sequnce length --> batch size, n_head, x seq len
    mask = (seq != pad_index).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq):
    seq_len = seq.size(-1)
    # matrix size, 1,1, seq len, seq len
    mask = torch.tril(torch.ones(seq_len, seq_len, device=seq.device)).bool().unsqueeze(0).unsqueeze(0)
    return mask


#   scores:          causal mask:         masked_fill后:
#   [[0.5, 0.3, 0.2],   [[1, 0, 0],      [[0.5, -inf, -inf],
#    [0.1, 0.4, 0.6], ×  [1, 1, 0],  →    [0.1, 0.4,  -inf],
#    [0.3, 0.2, 0.8]]    [1, 1, 1]]       [0.3, 0.2,   0.8]]