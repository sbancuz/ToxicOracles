import os
import sys
from argparse import ArgumentParser, Namespace
import logging

import re

import json
import pickle
import bz2

from joblib import Parallel, delayed, parallel_backend

from nltk.util import everygrams, padded_everygram_pipeline
from nltk.lm.preprocessing import flatten
from nltk.lm import KneserNeyInterpolated
from nltk.tokenize import word_tokenize
from numpy.f2py.rules import options

from transformers import AutoTokenizer

from datasets import load_dataset

from tqdm import tqdm


WIKITEXT2_SEP_REGEX = re.compile(r'( ?= [^=\n]+ = \n)')


def get_model_path(order, corpus, directory=None):
    file_name = f'{order}_gram__{corpus}.pbz2'
    return os.path.join(directory, file_name) if directory is not None else file_name



def save_model(lm, path):
    with bz2.BZ2File(path, 'wb') as f:
        pickle.dump(lm, f)


def load_model(path):
    with bz2.BZ2File(path, 'rb') as f:
        lm = pickle.load(f)

    return lm


def score_document(lm, document):
    return lm.perplexity(ngram for ngram in everygrams(document, min_len=lm.order, max_len=lm.order))


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


def load_book_corpus():
    train = (sample['text'] for sample in load_dataset(
        'bookcorpus/bookcorpus', streaming=True, split='train', trust_remote_code=True
    ))

    return train, None, None


def get_model(order, corpus, corpus_loader, model_dir):
    # Try to load the model
    model_path = get_model_path(order, corpus, directory=model_dir)
    try:
        logging.info(f'Attempting to load model from `{model_path}`')
        lm = load_model(model_path)
        logging.info(f'Model loaded')
    except FileNotFoundError:
        logging.info('Model loading failed fitting a new one on the current corpus')
        # If model is not available fit one on data set
        logging.info('Creating model')
        lm = KneserNeyInterpolated(order)
        logging.info('Model created')
        # Get data
        logging.info('Loading corpus')
        train_data, *_ = corpus_loader()
        logging.info('Corpus loaded')
        # Apply tokenization
        train_data = Parallel(verbose=2)(delayed(word_tokenize)(sample) for sample in train_data)
        # Prepare every-grams and vocabulary
        logging.info('Preparing data (this operation may take a while)')
        vocab = flatten(train_data)
        ngrams = (everygrams(sample, max_len=order) for sample in train_data)
        logging.info('Data preparation completed')
        # Fit model
        logging.info('Fitting model (this operation may take a while)')
        lm.fit(ngrams, vocab)
        logging.info('Model fitted')
        # Save model
        logging.info(f'Saving model at `{model_path}`')
        save_model(lm, model_path)
        logging.info('Model saved')

    return lm


def compute_perplexity(lm, data_path, lm_training_corpus):
    # Load prompt data
    logging.info(f'Loading prompt data from `{data_path}`')
    with open(data_path) as f:
        data = json.load(f)
    logging.info(f"Data loaded")
    # Iterate over prompts
    logging.info("Computing perplexity over data set")
    ppl = Parallel(verbose=2)(
        delayed(score_document)(prompt)
        for run in data['runs']
        for prompt in (
            run['initial']['prompt_from_dataset'],
            *(taken['input_prompt_for_generation'] for taken in run['taken'])
        )
    )
    logging.info("Perplexity computed")
    # Create results container
    logging.info("Reordering results")
    ppl_iterator = iter(ppl)
    results = {
        'results_file': data_path,
        'n-gram': {'order': lm.order, 'training_corpus': lm_training_corpus},
        'runs': [
            {
                'initial': {'prompt_from_dataset': run['initial']['prompt_from_dataset'], 'ppl': next(ppl_iterator)},
                'taken': [
                    {'input_prompt_for_generation': taken['input_prompt_for_generation'], 'ppl': next(ppl_iterator)}
                    for taken in run['taken']
                ]
            }
            for run in data['runs']
        ]
    }
    logging.info("Results reordered")
    # Save results
    file_name = os.path.basename(data_path)
    dir_path = os.path.dirname(data_path)
    results_path = os.path.join(dir_path, f'ppl_{lm.order}_gram__{lm_training_corpus}_{file_name}')
    logging.info(f'Saving results at `{results_path}`')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    logging.info(f'Results saved')


corpora = {'wikitext2': load_wikitext2, 'book_corpus': load_book_corpus}


def main(args: Namespace):
    # Start logging info
    logging.info('Script started')
    # Create checkpoint directory (if not exists)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        logging.info(f'Creating n-gram model directory at `{os.path.abspath(args.model_dir)}`')

    with parallel_backend('threading'):
        # Get n-gram language model
        lm = get_model(args.order, args.corpus, corpora[args.corpus], args.model_dir)
        logging.info(f'N-gram model ready for use')
        # Score perplexity on selected data
        compute_perplexity(lm, args.model_dir, args.corpus)
        logging.info(f'Perplexity computed')
    # Close script info
    logging.info("Script completed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='evotox_ngram_model_perplexity',
        description='Script to fit n-gram models on English data and compute perplexity (PPL) on given text'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--order',
        type=int,
        default=5,
        help="N-gram model order"
    )
    args_parser.add_argument(
        '--corpus',
        type=str,
        choices=[*corpora],
        required=True,
        help="Identifier of the corpus to be used to fit the n-gram model"
    )
    args_parser.add_argument(
        '--model_dir',
        type=str,
        default='ngrams',
        help="Path to the directory to save the n-gram models"
    )
    args_parser.add_argument(
        '--data_path',
        type=str,
        help="Path to the JSON file with the data generated by the evolutionary search"
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
