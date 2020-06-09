
import torch


class GreedySearch():

    def __init__(self, bos_id, eos_id, batch_size, segment_num=1, delete_id=None, pad=None, \
                 max_len_list=None, device=None):
        self.hypotheses = [[] for _ in range(batch_size)]
        self._batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        self.alive_seq = torch.full([batch_size, segment_num, 1], bos_id, \
                                     dtype=torch.long, device=device)
        self.is_seg_finished = torch.zeros([batch_size, segment_num], \
                                     dtype=torch.uint8, device=device)
        self.is_finished = None

        assert len(max_len_list) == batch_size * segment_num
        self.max_len_th = torch.tensor(max_len_list, dtype=torch.long, device=device)
        self.max_len_th = self.max_len_th.view(batch_size, segment_num)

        self.eos_id = eos_id
        self.delete_id = delete_id
        self.pad = pad
        self.batch_size = batch_size
        self.segment_num = segment_num
        self.device = device

        self.selected_indices = None
        self.done = False

    def search_one_step(self, token_score):
        topk_scores, topk_ids = token_score.max(dim=-1)
        topk_ids = topk_ids.masked_fill(self.is_seg_finished, self.pad)

        self.alive_seq = torch.cat(
            [self.alive_seq,
            topk_ids.view(-1, self.segment_num, 1)],
            dim=-1)

        len_exceed = self.max_len_th <= (self.alive_seq.size(-1) - 1)
        cur_seg_finished = topk_ids.eq(self.eos_id) | len_exceed
        if self.delete_id is not None:
            delete_flag = topk_ids.eq(self.delete_id)
            cur_seg_finished = cur_seg_finished | delete_flag

        self.is_seg_finished = self.is_seg_finished | cur_seg_finished
        self.is_finished = self.is_seg_finished.all(dim=1)

    def update_finished(self):
        finished = self.is_finished.nonzero().view(-1)
        for i in finished:
            b = self._batch_offset[i]
            self.hypotheses[b].append((None, self.alive_seq[i, :, 1:]))

        self.done = self.is_seg_finished.all()
        if self.done:
            return

        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        self.is_seg_finished = self.is_seg_finished[is_alive]
        self._batch_offset = self._batch_offset[is_alive]
        self.max_len_th = self.max_len_th[is_alive]
        self.selected_indices = is_alive.nonzero().view(-1)

    def get_final_results(self):
        return self.hypotheses
