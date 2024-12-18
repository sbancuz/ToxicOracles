import sys
from argparse import ArgumentParser, Namespace
import logging

import random

import re
import nltk

from datasets import load_dataset


RANDOM_SEED = 2307
WIKITEXT2_SEP_REGEX = re.compile(r'( ?= [^=\n]+ = \n)')


def preprocess_wikitext2_data(data):
    data = '\n'.join(sample['text'] for sample in data)
    data = WIKITEXT2_SEP_REGEX.sub(r'<|separator|>\1', data).strip()
    _, *data = data.split('<|separator|>')

    return data


def load_wikitext2():
    train = preprocess_wikitext2_data(load_dataset('wikitext', 'wikitext-2-raw-v1', split='train'))
    validation = preprocess_wikitext2_data(load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation'))
    test = preprocess_wikitext2_data(load_dataset('wikitext', 'wikitext-2-raw-v1', split='test'))

    return train, validation, test


def load_book_corpus_full_books():
    train = (sample['text'] for sample in load_dataset(
        # 'lucadiliello/bookcorpusopen', streaming=True, split='train', trust_remote_code=True
        'lucadiliello/bookcorpusopen', split='train', trust_remote_code=True
    ))

    return train, None, None


def load_book_corpus():
    train = (sample['text'] for sample in load_dataset(
        # 'bookcorpus/bookcorpus', streaming=True, split='train', trust_remote_code=True
        'bookcorpus/bookcorpus', split='train', trust_remote_code=True
    ))

    return train, None, None


corpora = {
    'wikitext2': load_wikitext2,
    'book_corpus': load_book_corpus_full_books,
    'book_corpus_sentences': load_book_corpus
}


def main(args: Namespace):
    # Start logging info
    logging.info('Script started')
    # Dowload tokeniser
    # nltk.download('punkt_tab')
    # logging.info('NLTK tools prepared')
    # Get corpus loader
    corpus_loader = corpora[args.corpus]
    logging.info('Corpus loader prepared')
    # Get data
    logging.info('Loading corpus')
    train_corpus, *_ = corpus_loader()
    logging.info('Corpus loaded')
    # Iterate over documents printing their content
    logging.info('Iterating over documents')
    for document in train_corpus:
        if random.random() < args.sample_size:  # Possibly apply down-sampling
            print(' '.join(nltk.word_tokenize(document)).lower())
    logging.info('Documents iteration completed')
    # Close script info
    logging.info('Script completed successfully')

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='evotox_ngram_model_perplexity',
        description='Script to fit n-gram models on English data and compute perplexity (PPL) on given text'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--corpus',
        type=str,
        choices=[*corpora],
        required=True,
        help="Identifier of the corpus to be used to yield data"
    )
    args_parser.add_argument(
        '--sample_size',
        type=float,
        default=1.0,
        required=False,
        help="Percentage of the corpus to use"
    )
    # Set random seed
    random.seed(RANDOM_SEED)
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
