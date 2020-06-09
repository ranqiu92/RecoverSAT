
import time
import itertools
import logging

import torch

from recoversat import RecoverSAT
from beamsearch import BeamSearch
from greedysearch import GreedySearch
from data import convert_to_tensor, get_reverse_dict


logger = logging.getLogger(__name__)

MAX_LENGTH_DIF = 50


class Translator():

    def __init__(self, model, src_vocab, tgt_vocab, batch_size, beam_size=1, device=None):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.reverse_tgt_vocab = get_reverse_dict(tgt_vocab)
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.device = device

    def translate(self, data):
        result = {'prediction': [], 'raw_prediction': []}

        all_preds = []
        start_time = time.time()
        for pos in range(0, len(data), self.batch_size):
            batch = list(itertools.islice(data, pos, pos + self.batch_size))

            if isinstance(self.model, RecoverSAT):
                batch_pred = self.recover_nat_translate_batch(batch)
            else:
                if self.beam_size == 1:
                    batch_pred = self.greedy_translate_batch(batch)
                else:
                    batch_pred = self.translate_batch(batch)
            all_preds.extend(batch_pred)

        end_time = time.time()
        logger.info('Total decoding time: %f' % (end_time - start_time))

        for prediction in all_preds:
            if not isinstance(self.model, RecoverSAT) and self.beam_size > 1:
                raw_pred = prediction[0][1].tolist()
                pred = raw_pred
            else:
                segment_list = prediction[0][1].tolist()

                pred, raw_pred = [], []
                for segment in segment_list:
                    seg_len = len(segment)
                    for j in range(seg_len):
                        if segment[seg_len - j - 1] != self.tgt_vocab['<pad>']:
                            break
                    segment = segment[: seg_len - j]

                    raw_pred.extend(segment)
                    if not isinstance(self.model, RecoverSAT) or segment[-1] != self.tgt_vocab['<delete>']:
                        pred.extend(segment)

            result['prediction'].append(pred)
            result['raw_prediction'].append(raw_pred)

        def _to_sentence(seq):
            raw_sentence = [self.reverse_tgt_vocab[id] for id in seq]
            sentence = " ".join(raw_sentence)
            return sentence

        def _to_cleaned_sentence(seq):
            raw_sentence = [self.reverse_tgt_vocab[id] for id in seq if id != self.tgt_vocab['<eos>']]
            sentence = " ".join(raw_sentence)
            return sentence

        result['raw_prediction'] = [_to_sentence(raw_pred) for raw_pred in result['raw_prediction']]
        result['prediction'] = [_to_cleaned_sentence(pred) for pred in result['prediction']]

        return result

    def recover_nat_translate_batch(self, batch):
        self.model.eval()

        batch_tensor = convert_to_tensor(batch, self.src_vocab, device=self.device)
        src_seq, src_lens = batch_tensor[:2]
        batch_size = src_seq.size(0)

        memory_bank, memory_mask = self.model.encode(src_seq, src_lens)

        max_len_list = []
        for init_len in src_lens.tolist():
            avg_len = init_len // self.model.segment_num
            remain_num = init_len % self.model.segment_num
            cur_max_len_list = [avg_len + 1 if i < remain_num else avg_len for i in range(self.model.segment_num)]
            max_len_list.extend(cur_max_len_list)
        max_len_list = [length + 20 for length in max_len_list]

        searcher = GreedySearch(
            bos_id=self.tgt_vocab['<bos>'],
            eos_id=self.tgt_vocab['<eos>'],
            delete_id=self.tgt_vocab['<delete>'],
            pad=self.tgt_vocab['<pad>'],
            batch_size=batch_size,
            segment_num=self.model.segment_num,
            max_len_list=max_len_list,
            device=self.device)

        position_base = torch.zeros((batch_size, self.model.segment_num), dtype=torch.long)
        seg_id_base = torch.arange(0, self.model.segment_num).unsqueeze(0).to(self.device)

        max_len = max(max_len_list)
        for step in range(max_len):
            dec_input = searcher.alive_seq[:, :, -1]
            dec_input = dec_input.view(-1, self.model.segment_num)
            cur_bsz = dec_input.size(0)

            position = position_base + step
            position = position.expand(cur_bsz, -1)
            seg_id = seg_id_base.expand(cur_bsz, -1)

            _, logit = self.model.decode(dec_input, memory_bank, memory_mask, step=step, pos=position, seg_id=seg_id)
            searcher.search_one_step(logit)

            if searcher.is_finished.any():
                searcher.update_finished()
                if searcher.done:
                    break

                select_indices = searcher.selected_indices
                memory_bank = memory_bank.index_select(0, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        predictions = searcher.get_final_results()
        return predictions

    def greedy_translate_batch(self, batch):
        self.model.eval()

        batch_tensor = convert_to_tensor(batch, self.src_vocab, device=self.device)
        src_seq, src_lens = batch_tensor[:2]
        batch_size = src_seq.size(0)

        memory_bank, memory_mask = self.model.encode(src_seq, src_lens)

        max_len_list = [MAX_LENGTH_DIF + length for length in src_lens.tolist()]
        searcher = GreedySearch(
            bos_id=self.tgt_vocab['<bos>'],
            eos_id=self.tgt_vocab['<eos>'],
            pad=self.tgt_vocab['<pad>'],
            batch_size=batch_size,
            max_len_list=max_len_list,
            device=self.device)

        max_len = max(max_len_list)
        for step in range(max_len):
            dec_input = searcher.alive_seq[:, :, -1].view(-1, 1)
            _, logit = self.model.decode(dec_input, memory_bank, memory_mask, step=step)
            searcher.search_one_step(logit)

            if searcher.is_finished.any():
                searcher.update_finished()
                if searcher.done:
                    break

                select_indices = searcher.selected_indices
                memory_bank = memory_bank.index_select(0, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)

                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        predictions = searcher.get_final_results()
        return predictions

    def translate_batch(self, batch):
        self.model.eval()

        batch_tensor = convert_to_tensor(batch, self.src_vocab, device=self.device)
        src_seq, src_lens = batch_tensor[:2]
        batch_size = src_seq.size(0)

        memory_bank, memory_mask = self.model.encode(src_seq, src_lens)
        memory_bank = memory_bank.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2), memory_bank.size(3))
        memory_mask = memory_mask.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        memory_mask = memory_mask.contiguous().view(-1, memory_mask.size(2), memory_mask.size(3))

        max_len_list = [MAX_LENGTH_DIF + length for length in src_lens.tolist()]
        searcher = BeamSearch(
            bos_id=self.tgt_vocab['<bos>'],
            eos_id=self.tgt_vocab['<eos>'],
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_len_list=max_len_list,
            device=self.device)

        max_len = max(max_len_list)
        for step in range(max_len):
            dec_input = searcher.alive_seq[:, -1].view(-1, 1)
            _, logit = self.model.decode(dec_input, memory_bank, memory_mask, step=step)
            log_prob = logit.squeeze(1).log_softmax(dim=-1)

            searcher.search_one_step(log_prob)
            any_beam_is_finished = searcher.is_finished.any()
            if any_beam_is_finished:
                searcher.update_finished()
                if searcher.done:
                    break

            select_indices = searcher.selected_indices

            if any_beam_is_finished:
                memory_bank = memory_bank.index_select(0, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        predictions = searcher.get_final_results()
        return predictions
