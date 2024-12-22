import os
import sys
from argparse import ArgumentParser, Namespace
import logging

import re

import numpy as np

import json
from kenlm import Model

from joblib import Parallel, delayed, parallel_backend

from nltk.tokenize import word_tokenize, sent_tokenize
from textstat import flesch_reading_ease

from evolutionary import Archive

from typing import Optional, Tuple, Pattern, List, Dict

KENLM_MODEL_NAME_REGEX: Pattern[str] = re.compile(r'(\d+)-gram\.(\w+)\.arpa')


def parse_model_path(model_path: str) -> Optional[Tuple[int, str]]:
    model_name = os.path.basename(model_path)
    match = KENLM_MODEL_NAME_REGEX.match(model_name)
    if match is not None:
        order, corpus = match.groups()
        order = int(order)
        return order, corpus
    else:
        return None


def score_document(lm: Model, document: str, sentence_split: bool) -> float:
    if '</newprompt>' in document:
        document, *_ = document.split('</newprompt>')
    if sentence_split:
        return np.mean(
            [lm.perplexity(' '.join(word_tokenize(sentence)).lower()) for sentence in sent_tokenize(document)]
        )
    else:
        return lm.perplexity(' '.join(word_tokenize(document)).lower())

def score_reading_ease(document: str) -> float:
    if '</newprompt>' in document:
        document, *_ = document.split('</newprompt>')
    return flesch_reading_ease(document)


def main(args: Namespace):
    # Start logging info
    logging.info('Script started')
    # Build results path
    model_name: str = os.path.basename(args.model)
    file_name: str = os.path.basename(args.data_path)
    dir_path: str = os.path.dirname(args.data_path)
    for d in ['perplexity', model_name.replace('.', '_')]:
        dir_path = os.path.join(dir_path, d)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    results_path: str = os.path.join(dir_path, f'ppl_{file_name}')
    # Check if results have already been computed
    if os.path.exists(results_path):
        logging.info(f'Results already computed and saved at `{results_path}`, skipping computation')

    else:
        # Load prompt data
        logging.info(f'Loading prompt data from `{args.data_path}`')
        with open(args.data_path) as f:
            data: Dict = json.load(f)
        logging.info(f"Data loaded")
        # Get n-gram language model
        lm: Model = Model(args.model)
        logging.info(f'N-gram model ready for use')
        # Iterate over prompts
        logging.info("Computing perplexity over data set")
        with parallel_backend('threading'):
            ppl: List[float] = Parallel(verbose=2)(
                delayed(score_document)(lm, prompt, 'sentence' in model_name)
                for run in data['runs']
                for prompt in (
                    run['initial'][
                        'prompt_from_dataset' if 'prompt_from_dataset' in run['initial'] else 'promptFromDataset'
                    ],
                    *(taken['input_prompt_for_generation'] for taken in run['taken'])
                )
            )
            read_ease: List[float] = Parallel(verbose=2)(
                delayed(score_reading_ease)(prompt)
                for run in data['runs']
                for prompt in (
                    run['initial'][
                        'prompt_from_dataset' if 'prompt_from_dataset' in run['initial'] else 'promptFromDataset'
                    ],
                    *(taken['input_prompt_for_generation'] for taken in run['taken'])
                )
            )
        logging.info("Perplexity and reading ease computed")
        # Create results container
        logging.info("Reordering results")
        ppl_iterator = iter(ppl)
        read_ease_iterator = iter(read_ease)
        results = {
            'handle': os.path.splitext(file_name)[0],
            'config': Archive.from_dict(data).config.to_dict(),
            'results_file': args.data_path,
            'n-gram': dict(zip(('order', 'training_corpus'), parse_model_path(args.model))),
            'model': model_name,
            'runs': [
                {
                    'initial': {
                        'prompt_from_dataset': run['initial'][
                            'prompt_from_dataset' if 'prompt_from_dataset' in run['initial'] else 'promptFromDataset'
                        ],
                        'ppl': next(ppl_iterator),
                        'reading_ease': next(read_ease_iterator),
                        'score': run['initial']['score']
                    },
                    'taken': [
                        {
                            'input_prompt_for_generation': taken['input_prompt_for_generation'],
                            'ppl': next(ppl_iterator),
                            'reading_ease': next(read_ease_iterator),
                            'score': taken['score']
                        }
                        for taken in run['taken']
                    ]
                }
                for run in data['runs']
            ]
        }
        logging.info("Results reordered")
        # Save results
        logging.info(f'Saving results at `{results_path}`')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        logging.info(f'Results saved')
    # Close script info
    logging.info("Script completed successfully")

    return 0


if __name__ == "__main__":
    # TODO merge into single PPL computation (including LLM one)
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='evotox_ngram_model_perplexity',
        description='Script to compute the perplexity of a given n-gram language model on a given set of prompts '
                    'generated with the evolutionary search'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Path to the n-gram model to use"
    )
    args_parser.add_argument(  # TODO convert to list of strings and handle file list rather than single file
        '--data_path',
        type=str,
        required=True,
        help="Path to the JSON file with the data generated by the evolutionary search"
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
