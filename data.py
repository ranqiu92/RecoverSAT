
import copy
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable


def load_vocab(file_path):
    vocab = OrderedDict()

    special_symbols = ['<pad>', '<bos>', '<eos>', '<unk>', '<delete>']
    for i, symb in enumerate(special_symbols):
        vocab[symb] = i

    idx = len(vocab)
    with open(file_path, encoding='utf8') as f:
        for line in f:
            w = line.strip()
            if w in vocab.keys():
                continue
            vocab[w] = idx
            idx += 1
    return vocab


def get_reverse_dict(dictionary):
    reverse_dict = {dictionary[k] : k for k in dictionary.keys()}
    return reverse_dict


def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    return padded_seqs


def get_segmented_sequence(seq_batch, seg_num, bos_id, eos_id, rand_dividing_prob=0, non_postfix_list=None,
                           redundant_prob=0., delete_id=None, padding=None):
    if seg_num <= 1:
        return seq_batch

    if redundant_prob > 0:
        assert delete_id is not None
        assert padding is not None

    pos_batch, seg_id_batch = [], []
    seg_input_batch, seg_label_batch, seg_lens_batch = [], [], []

    for seq, non_postfix in zip(seq_batch, non_postfix_list):
        rand_dividing_flag = np.random.sample() < rand_dividing_prob
        redundant_flag = np.random.sample() < redundant_prob
        cur_seg_num = seg_num - 1 if redundant_flag else seg_num

        seq_len = len(seq)

        valid_split_idx = [i for i in range(seq_len) if i not in non_postfix]
        valid_split_num = len(valid_split_idx)
        if valid_split_num < cur_seg_num:
            mid_idx_list = valid_split_idx
        else:
            mid_idx_list = []
            if rand_dividing_flag:
                random_idx_list = np.random.choice(valid_split_num, cur_seg_num - 1, replace=False)
                random_idx_list = random_idx_list.tolist()

                mid_idx_list = [valid_split_idx[ind] for ind in random_idx_list]
                mid_idx_list = sorted(mid_idx_list)
            else:
                avg_len = seq_len // cur_seg_num
                avg_len_list = [avg_len + 1 if i < seq_len % cur_seg_num else avg_len for i in range(cur_seg_num)]

                start = 0
                cur_avg_split_idx = -1
                for i in range(cur_seg_num):
                    cur_avg_split_idx += avg_len_list[i]

                    pre_dist = seq_len
                    for j in range(start, valid_split_num):
                        cur_split_idx = valid_split_idx[j]
                        dist = abs(cur_avg_split_idx - cur_split_idx)
                        if dist >= pre_dist:
                            mid_idx_list.append(valid_split_idx[j - 1])
                            start = j
                            break

        mid_idx_list = mid_idx_list + [seq_len - 1] * (cur_seg_num - len(mid_idx_list))

        if redundant_flag:
            candidate_seg_list = [0] + [i for i in range(1, cur_seg_num) if mid_idx_list[i] != mid_idx_list[i-1]]
            selected_seg_idx = np.random.choice(candidate_seg_list, 1).tolist()[0]

        pos, seg_id = [], []
        seg_input, seg_label, seg_lens = [], [], []

        pre_split_idx, cur_seg_id = 0, 0
        for i, split_idx in enumerate(mid_idx_list):
            seg_input = seg_input + [bos_id] + seq[pre_split_idx : split_idx + 1]
            seg_label = seg_label + seq[pre_split_idx : split_idx + 1] + [eos_id]

            cur_seg_len = split_idx - pre_split_idx + 2
            seg_id.extend([cur_seg_id] * cur_seg_len)
            seg_lens.append(cur_seg_len)
            pos.extend(list(range(cur_seg_len)))
            cur_seg_id += 1

            # inject pseudo redundant segment
            if redundant_flag and i == selected_seg_idx:
                repeat_len = np.random.choice(split_idx - pre_split_idx + 1, 1).tolist()[0] + 1
                pseudo_segment = seq[pre_split_idx : pre_split_idx + repeat_len]
                pseudo_segment_lbl = [padding] * repeat_len

                seg_input = seg_input + [bos_id] + pseudo_segment
                seg_label = seg_label + pseudo_segment_lbl + [delete_id]

                seg_id.extend([cur_seg_id] * (repeat_len + 1))
                seg_lens.append(repeat_len + 1)
                pos.extend(list(range(repeat_len + 1)))

                cur_seg_id += 1

            pre_split_idx = min(split_idx + 1, seq_len)

        seg_input_batch.append(seg_input)
        seg_label_batch.append(seg_label)
        seg_lens_batch.append(seg_lens)
        seg_id_batch.append(seg_id)
        pos_batch.append(pos)

    return seg_input_batch, seg_label_batch, seg_id_batch, pos_batch, seg_lens_batch


def convert_to_tensor(batch, src_vocab, tgt_vocab=None, seg_num=1, rand_dividing_prob=0., redundant_prob=0., device=None, is_training=False):
    src_pad = src_vocab['<pad>']

    src_seq = [sample['src_tokens'] for sample in batch]
    src_lens = [len(seq) for seq in src_seq]
    src_lens = torch.LongTensor(src_lens)
    padded_src_seq = padded_sequence(src_seq, src_pad)

    if is_training:
        tgt_pad = tgt_vocab['<pad>']
        tgt_bos = tgt_vocab['<bos>']
        tgt_eos = tgt_vocab['<eos>']
        tgt_delete_id = tgt_vocab['<delete>']

        tgt_seq = [sample['tgt_tokens'] for sample in batch]

        if seg_num > 1:
            non_postfix_list = [sample['non_postfix_list'] for sample in batch]
            tgt_segment_in, tgt_segment_out, seg_id, tgt_pos, seg_lens = get_segmented_sequence(tgt_seq, seg_num, \
                    bos_id=tgt_bos, eos_id=tgt_eos, rand_dividing_prob=rand_dividing_prob, non_postfix_list=non_postfix_list, \
                    redundant_prob=redundant_prob, delete_id=tgt_delete_id, padding=tgt_pad)
            input_tgt, label = tgt_segment_in, tgt_segment_out

            padded_tgt_seq = padded_sequence(input_tgt, tgt_pad)
            padded_label = padded_sequence(label, tgt_pad)
            padded_seg_id = padded_sequence(seg_id, seg_num - 1)
            padded_tgt_pos = padded_sequence(tgt_pos, 100)

            batch = [padded_src_seq, padded_tgt_seq, padded_label, padded_seg_id]
            batch = [Variable(torch.LongTensor(item), requires_grad=False) for item in batch]
            if device:
                batch = [item.to(device) for item in batch]
            src_seq, tgt_seq, label, seg_id = batch

            tgt_pos = Variable(torch.LongTensor(padded_tgt_pos), requires_grad=False)
            seg_lens = Variable(torch.LongTensor(seg_lens), requires_grad=False)
            return src_seq, src_lens, tgt_seq, label, seg_id, tgt_pos, seg_lens

        else:
            input_tgt = [[tgt_bos] + seq for seq in tgt_seq]
            label = [seq + [tgt_eos] for seq in tgt_seq]

            padded_tgt_seq = padded_sequence(input_tgt, tgt_pad)
            padded_label = padded_sequence(label, tgt_pad)

            batch = [padded_src_seq, padded_tgt_seq, padded_label]
            batch = [Variable(torch.LongTensor(item), requires_grad=False) for item in batch]
            if device:
                batch = [item.to(device) for item in batch]
            src_seq, tgt_seq, label = batch
            return src_seq, src_lens, tgt_seq, label
    else:
        src_seq = Variable(torch.LongTensor(padded_src_seq))
        if device:
            src_seq = src_seq.to(device)
        return src_seq, src_lens


def convert_word_to_id(sent, vocab):
    unk = vocab['<unk>']
    w_list = [w for w in sent.strip().split() if w]
    tokens = [vocab.get(w, unk) for w in w_list]
    return tokens


def load_data(src_file, src_vocab, tgt_file=None, tgt_vocab=None):
    with open(src_file, encoding='utf8') as f:
        src_sent_list = f.readlines()

    if tgt_file:
        with open(tgt_file, encoding='utf8') as f:
            tgt_sent_list = f.readlines()
        assert len(src_sent_list) == len(tgt_sent_list)

    for i, src_sent in enumerate(src_sent_list):
        sample = {
            'src_tokens': None,
            'tgt_tokens': None,
            'non_postfix_list': None
        }

        src_tokens = convert_word_to_id(src_sent, src_vocab)
        sample['src_tokens'] = src_tokens

        if tgt_file:
            tgt_sent = tgt_sent_list[i]
            tgt_tokens = convert_word_to_id(tgt_sent, tgt_vocab)
            non_postfix_list = [j for j in range(len(tgt_sent)) if tgt_sent[j].endswith('@@')]

            sample['tgt_tokens'] = tgt_tokens
            sample['non_postfix_list'] = non_postfix_list

        yield sample


def parallel_data_len(sample):
    src_len = len(sample['src_tokens']) if sample['src_tokens'] else 0
    tgt_len = len(sample['tgt_tokens']) if sample['tgt_tokens'] else 0
    return max(src_len, tgt_len)


def cluster_fn(data, bucket_size, len_fn):
    def cluster(id):
        return len_fn(data[id]) // bucket_size
    return cluster


def token_number_batcher(data, max_token_num, len_fn, bucket_size=3):
    sample_ids = list(range(len(data)))
    np.random.shuffle(sample_ids)
    sample_ids = sorted(sample_ids, key=cluster_fn(data, bucket_size, len_fn))

    total_len = 0
    sample_lens = []
    batch, batch_list = [], []
    for sample_id in sample_ids:
        batch.append(sample_id)
        length = len_fn(data[sample_id])
        sample_lens.append(length)
        total_len += length

        if total_len >= max_token_num:
            batch_list.append(batch)
            total_len = 0
            sample_lens = []
            batch = []

    if batch:
        batch_list.append(batch)
    np.random.shuffle(batch_list)
    for batch in batch_list:
        yield [data[id] for id in batch]
