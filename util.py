
import random
import numpy as np

import torch
import torch.nn as nn


class Transformer_LR_Schedule():

    def __init__(self, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step += 1
        scale = self.model_size ** -0.5
        scale *= min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return scale


class Linear_LR_Schedule():

    def __init__(self, initial_lr, final_lr, total_steps):
        self.initial_lr = initial_lr
        self.slope = (initial_lr - final_lr) / total_steps

    def __call__(self, step):
        scale = 1.0 - step * self.slope / self.initial_lr
        scale = max(scale, 0.)
        return scale


def set_random_seed(seed, is_cuda):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        torch.cuda.manual_seed(seed)

    return seed


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def segment_mask(lengths, max_len, seg_num, seg_lens):
    bsz = lengths.size(0)

    lengths = lengths.view(-1)
    max_len = max_len or lengths.max()
    seq_mask = sequence_mask(lengths, max_len)
    seq_mask = seq_mask.unsqueeze(1) * seq_mask.unsqueeze(-1)

    seg_lens = seg_lens.to(lengths.device)
    cumlen = torch.cumsum(seg_lens, dim=-1)
    zero_tensor = torch.zeros((bsz, 1), dtype=cumlen.dtype, device=cumlen.device)
    seg_start = torch.cat((zero_tensor, cumlen[:, :-1]), dim=-1)
    seg_end = cumlen

    len_range = torch.arange(0, max_len).to(lengths.device)
    len_range = len_range.view(1, 1, max_len).expand(bsz, seg_num, -1)

    upper_bound_mask = len_range < seg_end.unsqueeze(-1)
    lower_bound_mask = len_range >= seg_start.unsqueeze(-1)

    gen_order = len_range - seg_start.unsqueeze(-1)
    gen_order = gen_order * upper_bound_mask.long() * lower_bound_mask.long()
    gen_order = torch.sum(gen_order, dim=1)

    mask = gen_order.unsqueeze(-1) >= gen_order.unsqueeze(1)
    mask = mask * seq_mask

    return mask


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
