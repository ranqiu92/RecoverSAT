
import torch
import torch.nn.functional as F


class BeamSearch():

    def __init__(self, bos_id, eos_id, batch_size, device, beam_size=4, max_len_list=None):
        self.hypotheses = [[] for _ in range(batch_size)]
        self.alive_seq = torch.full([batch_size * beam_size, 1], bos_id, dtype=torch.long, device=device)
        self.topk_scores = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size)
        self.topk_ids = torch.empty((batch_size, beam_size), dtype=torch.long, device=device)
        self.is_finished = torch.zeros([batch_size, beam_size], dtype=torch.uint8, device=device)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        self._beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)

        assert len(max_len_list) == batch_size
        self.max_len_th = torch.tensor(max_len_list, dtype=torch.long, device=device)
        self.max_len_th = self.max_len_th.unsqueeze(-1).expand(-1, beam_size)

        self.eos_id = eos_id
        self.batch_size = batch_size
        self.beam_size = beam_size

        self.device = device
        self.selected_indices = None
        self.done = False
        self.alpha = 0.6

    def _length_normalized_score(self, score, length, alpha):
        return score / ((length + 5.) / (1. + 5.)) ** alpha

    def search_one_step(self, log_prob):
        vocab_size = log_prob.size(-1)
        cur_bsz = log_prob.size(0) // self.beam_size

        scores = self.topk_scores.view(-1, 1) + log_prob
        self.topk_scores, self.topk_ids = torch.topk(
                                    scores.view(cur_bsz, -1),
                                    self.beam_size,
                                    dim=-1)

        self.selected_indices = torch.div(self.topk_ids, vocab_size) \
                              + self._beam_offset[:cur_bsz].unsqueeze(1)
        self.selected_indices = self.selected_indices.view(cur_bsz * self.beam_size)

        self.topk_ids = torch.fmod(self.topk_ids, vocab_size)
        self.alive_seq = torch.cat(
                    [self.alive_seq.index_select(0, self.selected_indices),
                    self.topk_ids.view(-1, 1)],
                    dim=1)

        len_exceed = self.max_len_th <= (self.alive_seq.size(-1) - 1)
        self.is_finished = self.topk_ids.eq(self.eos_id) | len_exceed        

    def update_finished(self):
        cur_bsz = self.alive_seq.size(0) // self.beam_size
        length = self.alive_seq.size(-1) - 1
        predictions = self.alive_seq.view(cur_bsz, self.beam_size, -1)

        normalized_score = self._length_normalized_score(self.topk_scores, length, self.alpha)

        self.topk_scores = self.topk_scores.masked_fill(self.is_finished, -1e10)
        best_alive_score, _ = torch.max(
            self._length_normalized_score(self.topk_scores, self.max_len_th.float(), self.alpha),
            dim=1)

        non_finished = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            for j in finished_hyp:
                self.hypotheses[b].append((float(normalized_score[i, j]), predictions[i, j, 1:]))

            if len(self.hypotheses[b]) >= self.beam_size:
                self.hypotheses[b] = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)[:self.beam_size]
                worst_finished_score = self.hypotheses[b][-1][0]
                if worst_finished_score < best_alive_score[i]:
                    non_finished.append(i)
            else:
                non_finished.append(i)

        if len(non_finished) == 0:
            self.done = True
            return

        non_finished = torch.tensor(non_finished, device=self.device)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.selected_indices = self.selected_indices.view(-1, self.beam_size).index_select(0, non_finished).view(-1)
        self.alive_seq = predictions.index_select(0, non_finished).view(-1, self.alive_seq.size(-1))
        self.max_len_th = self.max_len_th.index_select(0, non_finished)

    def get_final_results(self):
        length = self.alive_seq.size(-1) - 1
        normalized_score = self._length_normalized_score(self.topk_scores, length, self.alpha)

        self.alive_seq = self.alive_seq.view(-1, self.beam_size, self.alive_seq.size(-1))
        unfinished = ~self.alive_seq[:, :, -1].eq(self.eos_id)
        for i in range(unfinished.size(0)):
            b = self._batch_offset[i]
            unfinished_hyp = unfinished[i].nonzero().view(-1)
            for j in unfinished_hyp:
                self.hypotheses[b].append((float(normalized_score[i, j]), self.alive_seq[i, j, 1:]))

        for b in range(self.batch_size):
            if len(self.hypotheses[b]) > self.beam_size:
                self.hypotheses[b] = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)[:self.beam_size]

        return self.hypotheses
