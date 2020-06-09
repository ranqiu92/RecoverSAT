
import os
import argparse
import logging
from distutils.util import strtobool

import torch

from translator import Translator
from data import load_vocab, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Predicting")

    parser.add_argument("--model_path", required=True, type=str,
                        help="The path of the model parameter file")

    parser.add_argument("--batch_size", default=64, type=int,
                        help="The minibatch size")
    parser.add_argument("--beam_size", default=4, type=int,
                        help="The beam width of the beam search")

    parser.add_argument("--input_file", required=True, type=str,
                        help="The path of the input file")
    parser.add_argument("--output_file", required=True, type=str,
                        help="The path of file to save the predictions")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="The path of the vocabulary")

    parser.add_argument("--share_vocab", default=True, type=strtobool,
                        help="Whether to share the src/tgt vocabs")
    parser.add_argument("--src_vocab_path", default=None, type=str,
                        help="The path of the src vocabulary")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="The path of the tgt vocabulary")

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()

    # load data
    if args.share_vocab:
        vocab = load_vocab(args.vocab_path)
        src_vocab = vocab
        tgt_vocab = vocab
        vocab_size = len(vocab)
        logger.info("Load vocabulary from %s, vocabulary size: %d" % (args.vocab_path, vocab_size))
    else:
        src_vocab = load_vocab(args.src_vocab_path)
        tgt_vocab = load_vocab(args.tgt_vocab_path)
        src_vocab_size, tgt_vocab_size = len(src_vocab), len(tgt_vocab)
        logger.info("Load src vocabulary from %s, vocabulary size: %d" % (args.src_vocab_path, src_vocab_size))
        logger.info("Load tgt vocabulary from %s, vocabulary size: %d" % (args.tgt_vocab_path, tgt_vocab_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    use_gpu = n_gpu > 0

    model = torch.load(args.model_path)
    model.to(device)

    input_data = list(load_data(args.input_file, src_vocab))
    logger.info("Load dataset with %d samples from %s" % (len(input_data), args.input_file))

    translator = Translator(
        model=model,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        device=device)

    with torch.no_grad():
        results = translator.translate(input_data)
    prediction = results['prediction']
    content = '\n'.join(prediction) + '\n'
    with open(args.output_file, 'w', encoding='utf8') as f:
        f.write(content)

    logger.info('%d samples have been translated.' % len(prediction))


if __name__ == '__main__':
    main()
