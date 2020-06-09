
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from util import sequence_mask, init_weights


def smooth_loss(log_prob, label, pad, eps=0.15):
    mask = label == pad

    nclass = log_prob.size(1)
    smoothed_label = torch.zeros_like(log_prob)
    smoothed_label = smoothed_label + eps / (nclass - 1)

    class_range = torch.arange(0, nclass).to(log_prob.device)
    selected_mask = class_range.view(1, nclass, 1).expand_as(log_prob) \
                    == label.unsqueeze(1).expand_as(log_prob)
    smoothed_label = smoothed_label.masked_fill(selected_mask, 1 - eps)
    smoothed_label = smoothed_label.masked_fill(mask.unsqueeze(1), 0.)

    loss = torch.sum(-log_prob * smoothed_label)

    return loss


class PositionEmbedding(nn.Module):

    def __init__(self, emb_size, max_timescale=1.0e4):
        super(PositionEmbedding, self).__init__()
        self.emb_size = emb_size
        self.max_timescale = max_timescale

    def forward(self, length=None, step=None):
        assert length is not None or step is not None

        if length is not None:
            pos = torch.arange(0., length).unsqueeze(-1)
        if isinstance(step, int):
            pos = torch.tensor([[step]], dtype=torch.float)
        elif step is not None:
            pos = step.unsqueeze(-1).float()

        dim = torch.arange(0., self.emb_size, 2.).unsqueeze(0) / self.emb_size
        dim = dim.to(pos.device)

        sin = torch.sin(pos / torch.pow(self.max_timescale, dim))
        cos = torch.cos(pos / torch.pow(self.max_timescale, dim))

        pos_emb = torch.stack((sin, cos), -1).view(pos.size(0), -1)
        pos_emb = pos_emb[:, :self.emb_size]

        return pos_emb


class FeedForward(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        mid = F.relu(self.layer_1(inputs))
        return self.layer_2(self.dropout(mid))


class MultiHeadedAttention(nn.Module):

    def __init__(self, head_num, hidden_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % head_num == 0
        self.head_size = hidden_size // head_num
        self.hidden_size = hidden_size
        self.head_num = head_num

        self.linear_keys = nn.Linear(hidden_size, head_num * self.head_size)
        self.linear_values = nn.Linear(hidden_size, head_num * self.head_size)
        self.linear_query = nn.Linear(hidden_size, head_num * self.head_size)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, mask=None, layer_cache=None, attn_type=None, mask_self=False):
        batch_size = key.size(0)
        head_size = self.head_size
        head_num = self.head_num

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_num, head_size) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_num * head_size)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key, value = shape(key), shape(value)

                if layer_cache["self_keys"] is not None:
                    key = torch.cat((layer_cache["self_keys"], key), dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat((layer_cache["self_values"], value), dim=2)

                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key, value = shape(key), shape(value)

                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key, value = shape(key), shape(value)
        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / head_size ** 0.5
        scores = torch.matmul(query, key.transpose(2, 3)).float()
        if mask is not None:
            if attn_type == "self" and layer_cache is not None:
                if layer_cache["mask"] is not None:
                    mask = torch.cat((layer_cache["mask"], mask), dim=2)
                layer_cache["mask"] = mask

            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e18)

        if mask_self and attn_type == "self" and layer_cache is None:
            diag_mask = torch.diagflat(
                torch.ones(
                    key.size(2),
                    dtype=torch.uint8,
                    device=scores.device))
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(diag_mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        context_original = torch.matmul(drop_attn, value)
        context = unshape(context_original)
        output = self.final_linear(context)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_size, heads, ffn_size, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, hidden_size, dropout=dropout)
        self.feed_forward = FeedForward(hidden_size, ffn_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm_1(inputs)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask, attn_type="self")
        mid = self.dropout(context) + inputs

        mid_norm = self.layer_norm_2(mid)
        ffn_out = self.feed_forward(mid_norm)
        output = self.dropout(ffn_out) + mid

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, hidden_size, heads, ffn_size, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                hidden_size, heads, ffn_size, dropout)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, input_lens):
        max_len = inputs.size(1)
        mask = ~sequence_mask(input_lens, max_len).unsqueeze(1).to(inputs.device)

        output = inputs
        for layer in self.encoder_layers:
            output = layer(output, mask)
        output = self.layer_norm(output)

        return output, mask


class TransformerDecoderLayer(nn.Module):

    def __init__(self, hidden_size, heads, ffn_size, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, hidden_size, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, hidden_size, dropout=dropout)
        self.feed_forward = FeedForward(hidden_size, ffn_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, mask_self=False):
        input_norm = self.layer_norm_1(inputs)
        query = self.self_attn(input_norm, input_norm, input_norm,
                                     mask=tgt_pad_mask,
                                     layer_cache=layer_cache,
                                     attn_type="self",
                                     mask_self=mask_self)
        query = self.dropout(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context")
        ffn_in = self.dropout(mid) + query
        normed_ffn_in = self.layer_norm_3(ffn_in)
        ffn_out = self.feed_forward(normed_ffn_in)
        out = self.dropout(ffn_out) + ffn_in

        return out


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, hidden_size, head_num, ffn_size, dropout, causal=True):
        super(TransformerDecoder, self).__init__()
        self.state = {}
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, head_num, ffn_size, dropout)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.causal = causal

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def forward(self, tgt, memory_bank, src_pad_mask, tgt_pad_mask, step=None, mask_self=False):
        dec_mask = tgt_pad_mask
        if self.causal and step is None:
            tgt_max_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_max_len, tgt_max_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_max_len, tgt_max_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        if step == 0:
            self._init_cache(memory_bank)

        output = tgt
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output = layer(
                output,
                memory_bank,
                src_pad_mask,
                dec_mask,
                layer_cache=layer_cache,
                mask_self=mask_self)
        output = self.layer_norm(output)

        return output

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(0)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.decoder_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            layer_cache["mask"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache


class Transformer(nn.Module):

    def __init__(self, enc_layers, dec_layers, hidden_size, head_num, ffn_size, src_emb_conf, tgt_emb_conf=None, \
                 dropout=0.1, use_label_smoothing=True, smooth_rate=0.15):
        super(Transformer, self).__init__()
        if tgt_emb_conf is None:
            self.embedding = self._init_embedding(src_emb_conf)
            self.src_embedding = self.embedding
            self.tgt_embedding = self.embedding
        else:
            self.src_embedding = self._init_embedding(src_emb_conf)
            self.tgt_embedding = self._init_embedding(tgt_emb_conf)
        self.pos_embedding = PositionEmbedding(hidden_size)
        self.encoder = TransformerEncoder(enc_layers, hidden_size, head_num, ffn_size, dropout)
        self.decoder = TransformerDecoder(dec_layers, hidden_size, head_num, ffn_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.use_label_smoothing = use_label_smoothing
        self.smooth_rate = smooth_rate
        self.apply(init_weights)

    def _init_embedding(self, emb_conf):
        vocab_size = emb_conf['vocab_size']
        emb_size = emb_conf['emb_size']
        padding_idx = emb_conf.get('padding_idx', None)
        return nn.Embedding(vocab_size, emb_size, padding_idx)

    def forward(self, src_seq, tgt_seq, src_lens, label, scoring=False):
        max_len = max(int(src_lens.max()), 1)
        src_seq = src_seq[:, :max_len]

        src_enc, src_mask = self.encode(src_seq, src_lens)
        tgt_dec, logit = self.decode(tgt_seq, src_enc, src_mask)

        log_prob = F.log_softmax(logit, dim=-1).transpose(1, 2)

        if scoring:
            loss_func = nn.NLLLoss(ignore_index=self.tgt_embedding.padding_idx, reduction='none')
            loss = loss_func(log_prob, label)
            return loss

        if self.use_label_smoothing:
            loss = smooth_loss(log_prob, label, pad=self.tgt_embedding.padding_idx, eps=self.smooth_rate)
        else:
            loss_func = nn.NLLLoss(ignore_index=self.tgt_embedding.padding_idx, reduction='sum')
            loss = loss_func(log_prob, label)

        return loss

    def embed(self, input, embedding, step=None):
        bsz, length = input.size()
        emb = embedding(input) * self.hidden_size ** 0.5
        pos_emb = self.pos_embedding(length, step=step).to(input.device)
        pos_emb = pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        emb = self.dropout(emb + pos_emb)
        return emb

    def encode(self, src, src_lens):
        src_emb = self.embed(src, self.src_embedding)
        src_enc, src_mask = self.encoder(src_emb, src_lens)
        return src_enc, src_mask

    def decode(self, tgt, src_enc, src_mask=None, step=None):
        padding_idx = self.tgt_embedding.padding_idx
        tgt_mask = tgt.data.eq(padding_idx).unsqueeze(1)
        tgt_emb = self.embed(tgt, self.tgt_embedding, step=step)

        tgt_dec = self.decoder(tgt_emb, src_enc, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask, step=step)
        logit = F.linear(tgt_dec, self.tgt_embedding.weight)

        return tgt_dec, logit
