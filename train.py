
import os
import random
import argparse
import logging
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu

from transformer import Transformer
from recoversat import RecoverSAT
from util import set_random_seed, Transformer_LR_Schedule, Linear_LR_Schedule
from data import load_vocab, load_data, parallel_data_len, \
    token_number_batcher, convert_to_tensor
from translator import Translator


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model_name", choices=["Transformer", "RecoverSAT"], type=str)

    parser.add_argument("--enc_layers", default=6, type=int,
                        help="The layer number of the encoder")
    parser.add_argument("--dec_layers", default=6, type=int,
                        help="The layer number of the encoder")
    parser.add_argument("--hidden_size", default=512, type=int,
                        help="The hidden size of the model")
    parser.add_argument("--ffn_size", default=512, type=int,
                        help="The size of the feedforward layer")
    parser.add_argument("--head_num", default=8, type=int,
                        help="The head number of the attention")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="The dropout probability")

    parser.add_argument("--segment_num", default=1, type=int,
                        help="The segment number")
    parser.add_argument("--rand_dividing_prob", default=1., type=float,
                        help="The probability of using random dividing method")
    parser.add_argument("--anneal_rand_dividing_prob", default=True, type=strtobool,
                        help="Whether to anneal the probability of using"
                        "random dividing method")
    parser.add_argument("--redundant_prob", default=0.5, type=float,
                        help="The probability of injecting redundant segment")

    parser.add_argument("--learning_rate", default=1., type=float,
                        help="The learning rate")
    parser.add_argument("--weight_decay", default=0., type=float,
                        help="The L2 weight decay rate")
    parser.add_argument("--lr_schedule", choices=["linear", "warmup"], type=str,
                        help="The learning rate schedule")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="The maximum batch number of lr warmup")
    parser.add_argument("--epoch_num", default=500, type=int,
                        help="The epoch number to train the model")
    parser.add_argument("--total_steps", default=250000, type=int,
                        help="The total steps of the training stage")
    parser.add_argument("--max_token_num", default=2048, type=int,
                        help="The maximum token number of each minibatch")
    parser.add_argument("--use_label_smoothing", default=True, type=strtobool,
                        help='Whether to use label smoothing or not')
    parser.add_argument("--smooth_rate", default=0.15, type=float,
                        help='The rate of label smoothing')

    parser.add_argument('--seed', default=None, type=int,
                        help="Random seed for initialization and sampling")
    parser.add_argument('--grad_accumulate_steps', default=1, type=int,
                        help='The step number of accumulating gradients before updating')

    parser.add_argument("--dataset", choices=["IWSLT16", "WMT16", "WMT14"], type=str)
    parser.add_argument("--train_src_file", required=True, type=str,
                        help="The path of the source training data")
    parser.add_argument("--train_tgt_file", required=True, type=str,
                        help="The path of the target training data")
    parser.add_argument("--valid_src_file", default=None, type=str,
                        help="The path of the source validation data")
    parser.add_argument("--valid_tgt_file", default=None, type=str,
                        help="The path of the target validation data")
    parser.add_argument("--valid_tgt_file_list", default=None, type=str,
                        help="The path list of the target validation data")

    parser.add_argument("--share_vocab", default=True, type=strtobool,
                        help="Whether to share the src/tgt vocabs")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="The path of the shared vocabulary")
    parser.add_argument("--src_vocab_path", default=None, type=str,
                        help="The path of the src vocabulary")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="The path of the tgt vocabulary")

    parser.add_argument("--init_encoder_path", default=None, type=str,
                        help="The path of the parameters to initialize NAT's encoder")
    parser.add_argument("--save_path", default="checkpoint", type=str,
                        help="The path to save the model")

    parser.add_argument("--log_period", default=50, type=int)
    parser.add_argument("--save_period", default=1000, type=int)

    args = parser.parse_args()
    return args


@torch.no_grad()
def evaluation(model, src_vocab, tgt_vocab, valid_data, reference, batch_size=64, beam_size=1, args=None, device=None):
    translator = Translator(
        model=model,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=batch_size,
        beam_size=beam_size,
        device=device)

    result = translator.translate(valid_data)
    prediction = result['prediction']
    prediction = [pred.replace('@@ ', '') for pred in prediction]
    prediction = [pred.split() for pred in prediction]
    bleu = corpus_bleu(reference, prediction) * 100.
    return bleu


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    if args.model_name == 'Transformer':
        args.segment_num = 1

    if args.dataset == 'IWSLT16':
        hparams = {'enc_layers': 5, 'dec_layers': 5, 'hidden_size': 278, 'ffn_size': 507, 'head_num': 2,
                   'lr_schedule': 'linear', 'learning_rate': 3e-4, 'total_steps': 250000, 'max_token_num': 2048}
    elif args.dataset == 'WMT14':
        hparams = {'enc_layers': 6, 'dec_layers': 6, 'hidden_size': 512, 'ffn_size': 512, 'head_num': 8,
                   'lr_schedule': 'warmup', 'learning_rate': 1, 'total_steps': 200000, 'max_token_num': 60000}
    else:
        hparams = {'enc_layers': 6, 'dec_layers': 6, 'hidden_size': 512, 'ffn_size': 512, 'head_num': 8,
                   'lr_schedule': 'warmup', 'learning_rate': 1, 'total_steps': 200000, 'max_token_num': 30000}
    args.__dict__.update(hparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    use_gpu = n_gpu > 0
    logger.info('n_gpu=%d' % n_gpu)

    if args.seed is None:
        seed = random.Random(None).randint(1, 100000)
    else:
        seed = args.seed
    set_random_seed(seed, use_gpu)
    logger.info("seed: {} ".format(seed))

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #----------------- Data Preparation ----------------#
    if args.share_vocab:
        vocab = load_vocab(args.vocab_path)
        src_vocab = vocab
        tgt_vocab = vocab
        vocab_size = len(vocab)
        src_emb_conf = {
            'vocab_size': vocab_size,
            'emb_size': args.hidden_size,
            'padding_idx': vocab['<pad>']
        }
        tgt_emb_conf = None
        logger.info("Load vocabulary from %s, vocabulary size: %d" % (args.vocab_path, vocab_size))
    else:
        src_vocab = load_vocab(args.src_vocab_path)
        tgt_vocab = load_vocab(args.tgt_vocab_path)
        src_vocab_size, tgt_vocab_size = len(src_vocab), len(tgt_vocab)
        logger.info("Load src vocabulary from %s, vocabulary size: %d" % (args.src_vocab_path, src_vocab_size))
        logger.info("Load tgt vocabulary from %s, vocabulary size: %d" % (args.tgt_vocab_path, tgt_vocab_size))

        src_emb_conf = {
            'vocab_size': src_vocab_size,
            'emb_size': args.hidden_size,
            'padding_idx': src_vocab['<pad>']
        }
        tgt_emb_conf = {
            'vocab_size': tgt_vocab_size,
            'emb_size': args.hidden_size,
            'padding_idx': tgt_vocab['<pad>']
        }

    tgt_padding_idx = tgt_vocab['<pad>']

    train_dataset = list(load_data(args.train_src_file, src_vocab, args.train_tgt_file, tgt_vocab))
    logger.info("Load training set with %d samples" % len(train_dataset))

    def split_str(string, sep=' '):
        return [substr for substr in string.split(sep) if substr]

    do_eval = False
    if args.valid_src_file is not None:
        do_eval = True

        valid_tgt_file_list = [args.valid_tgt_file]
        if args.valid_tgt_file_list:
            valid_tgt_file_list = eval(args.valid_tgt_file_list)

        valid_tgt_ref_list = []
        for valid_tgt_file in valid_tgt_file_list:
            with open(valid_tgt_file, encoding='utf8') as f:
                valid_tgt_sent = f.readlines()

            valid_tgt_ref = [split_str(sent.strip().replace('@@ ', '')) for sent in valid_tgt_sent]
            valid_tgt_ref_list.append(valid_tgt_ref)
        valid_tgt_ref = list(zip(*valid_tgt_ref_list))
        valid_tgt_ref = [list(ref) for ref in valid_tgt_ref]

        valid_dataset = list(load_data(args.valid_src_file, src_vocab))
        logger.info("Load validation set with %d samples" % len(valid_dataset))

    #--------------- Model Initialization ---------------#
    if args.model_name == 'Transformer':
        model = Transformer(
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            hidden_size=args.hidden_size,
            head_num=args.head_num,
            ffn_size=args.ffn_size,
            src_emb_conf=src_emb_conf,
            tgt_emb_conf=tgt_emb_conf,
            dropout=args.dropout,
            use_label_smoothing=args.use_label_smoothing,
            smooth_rate=args.smooth_rate)

    elif args.model_name == 'RecoverSAT':
        model = RecoverSAT(
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            hidden_size=args.hidden_size,
            head_num=args.head_num,
            ffn_size=args.ffn_size,
            src_emb_conf=src_emb_conf,
            tgt_emb_conf=tgt_emb_conf,
            eos_id=tgt_vocab['<eos>'],
            delete_id=tgt_vocab['<delete>'],
            segment_num=args.segment_num,
            dropout=args.dropout,
            use_label_smoothing=args.use_label_smoothing,
            smooth_rate=args.smooth_rate)

    else:
        assert False

    if args.init_encoder_path:
        init_model = torch.load(args.init_encoder_path)
        model.src_embedding.load_state_dict(init_model.src_embedding.state_dict())
        model.tgt_embedding.load_state_dict(init_model.tgt_embedding.state_dict())
        model.encoder.load_state_dict(init_model.encoder.state_dict())
        logger.info("Load pretrained embedding and encoder parameters from %s" % args.init_encoder_path)

    #--------------- Training Preparation --------------#
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(
        params=trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9)

    total_steps = args.total_steps
    if args.lr_schedule == 'warmup':
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=Transformer_LR_Schedule(
                model_size=args.hidden_size,
                warmup_steps=args.warmup_steps))
    elif args.lr_schedule == 'linear':
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=Linear_LR_Schedule(
                initial_lr=args.learning_rate,
                final_lr=1e-5,
                total_steps=total_steps))
    else:
        assert False, "Unrecognized learning rate schedule: %s" % args.lr_schedule

    #-------------------- Training ---------------------#
    if args.model_name == 'Transformer':
        beam_size_list = [4]
    else:
        beam_size_list = [1]

    best_bleu, best_ckp = dict(), dict()
    for beam_size in beam_size_list:
        key = 'b%d' % beam_size
        best_bleu[key] = 0
        best_ckp[key] = {'epoch': -1, 'batch': -1}

    if do_eval:
        eval_model = model.module if n_gpu > 1 else model

    assert args.grad_accumulate_steps >= 1
    token_num_per_batch = args.max_token_num // args.grad_accumulate_steps

    real_loss = 0.
    global_step, mini_step = 0, 0
    for epoch in range(args.epoch_num):
        for step, batch in enumerate(token_number_batcher(train_dataset, token_num_per_batch, parallel_data_len)):
            cur_rand_dividing_prob = args.rand_dividing_prob
            if args.anneal_rand_dividing_prob:
                cur_rand_dividing_prob = args.rand_dividing_prob * max(1. - global_step / total_steps, 0.)

            batch_tensor = convert_to_tensor(batch, src_vocab, tgt_vocab, seg_num=args.segment_num, \
                rand_dividing_prob=cur_rand_dividing_prob, redundant_prob=args.redundant_prob, device=device, is_training=True)
            src_seq, src_lens, tgt_seq, label = batch_tensor[:4]

            if args.model_name == 'RecoverSAT':
                seg_id, tgt_pos, seg_lens = batch_tensor[-3:]

            if args.model_name == 'Transformer':
                loss = model(src_seq, tgt_seq, src_lens, label)
                token_num = torch.clamp(torch.sum(label != tgt_padding_idx).float(), min=1.)
                loss = loss.sum() / token_num

            elif args.model_name == 'RecoverSAT':
                loss = model(src_seq, tgt_seq, src_lens, label, seg_id=seg_id, tgt_pos=tgt_pos, seg_lens=seg_lens)
                token_num = torch.clamp(torch.sum(label != tgt_padding_idx).float(), min=1.)
                loss = loss.sum() / token_num

            loss = loss / args.grad_accumulate_steps
            loss.backward()

            real_loss = real_loss + loss.item()
            mini_step += 1

            if mini_step % args.grad_accumulate_steps == 0:
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_period == 0:
                    log_info = "Epoch=%-3d batch=%-4d step=%-6d loss=%f" % (epoch, step + 1, global_step, real_loss)
                    logger.info(log_info)

                if do_eval and global_step % args.save_period == 0:
                    model.eval()
                    for beam_size in beam_size_list:
                        bleu = evaluation(eval_model, src_vocab, tgt_vocab, valid_dataset, valid_tgt_ref, \
                                                            beam_size=beam_size, args=args, device=device)
                        log_info = "Evaluation: beam_size=%d Epoch=%-3d batch=%-4d step=%-6d bleu=%.2f" \
                                   % (beam_size, epoch, step + 1, global_step, bleu)
                        logger.info(log_info)

                        key = 'b%d' % beam_size
                        if bleu > best_bleu[key]:
                            prev_model_file = os.path.join(save_path, 'b%d-epoch-%d-batch-%d.ckp' \
                                   % (beam_size, best_ckp[key]['epoch'], best_ckp[key]['batch']))
                            if os.path.exists(prev_model_file):
                                os.remove(prev_model_file)

                            best_ckp[key]['epoch'] = epoch
                            best_ckp[key]['batch'] = step + 1
                            best_bleu[key] = bleu

                            model_file = os.path.join(save_path, 'b%d-epoch-%d-batch-%d.ckp' % (beam_size, epoch, step + 1))
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save, model_file)

                        log_info = "Evaluation: beam_size=%d BEST_BLEU:%.2f BEST_CKP:epoch-%d-batch-%d" \
                                   % (beam_size, best_bleu[key], best_ckp[key]['epoch'], best_ckp[key]['batch'])
                        logger.info(log_info)
                    model.train()
                real_loss = 0
        if global_step >= total_steps:
            break

    if do_eval:
        for beam_size in beam_size_list:
            key = 'b%d' % beam_size
            log_info = "Final: DEV beam_size=%d BEST_BLEU:%.2f BEST_CKP:epoch-%d-batch-%d" \
                       % (beam_size, best_bleu[key], best_ckp[key]['epoch'], best_ckp[key]['batch'])
            logger.info(log_info)


if __name__ == '__main__':
    main()
