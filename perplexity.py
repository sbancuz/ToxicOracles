import os
import sys
from argparse import ArgumentParser, Namespace
import logging

from itertools import batched
from tqdm import tqdm
import json

import torch
from torchmetrics.functional.text import perplexity

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from evolutionary import Archive


@torch.no_grad()
def main(args: Namespace):
    # Start logging info
    logging.info("Script started")
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: `{device}`")
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype=torch.bfloat16,
        # attn_implementation='flash_attention_2',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        ),
        token=args.token
    )
    model.eval()
    logging.info(f"Model `{args.model}` loaded")
    tokeniser = AutoTokenizer.from_pretrained(args.model, token=args.token)
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
    logging.info("Tokeniser loaded")
    # Load data set
    with open(args.data_path) as f:
        data = json.load(f)
    logging.info(f"Data loaded from `{args.data_path}`")
    # Iterate over data set batches
    logging.info("Computing perplexity over data set")
    ppl = list()
    for batch in tqdm(batched((
            prompt
            for run in data['runs']
            for prompt in (
                run['initial']['prompt_from_dataset'], *(taken['input_prompt_for_generation'] for taken in run['taken'])
            )
    ), n=args.batch_size)):
        # Encode input
        input_encodings = tokeniser(batch, return_tensors='pt', padding=True).to(device)
        # Encode target output
        labels = input_encodings.input_ids.clone()
        labels[input_encodings.attention_mask == 0] = -100
        # Run data through model
        logits = model.forward(**input_encodings).logits
        # Shift logits to exclude the last element
        logits = logits[..., :-1, :].contiguous()
        # shift labels to exclude the first element
        labels = labels[..., 1:].contiguous()
        # Compute perplexity
        ppl.extend(perplexity(logits[i:i+1], labels[i:i+1], ignore_index=-100).view(1) for i in range(len(batch)))
    ppl = torch.cat(ppl, dim=0).cpu().tolist()
    logging.info("Done computing perplexity over data set")
    # Create results container
    ppl_iterator = iter(ppl)
    results = {
        'handle': os.path.splitext(os.path.basename(args.data_path))[0],
        'config': Archive.from_dict(data).config.to_dict(),
        'results_file': args.data_path,
        'model': args.model,
        'runs': [
            {
                'initial': {
                    'prompt_from_dataset': run['initial'][
                        'prompt_from_dataset' if 'prompt_from_dataset' in run['initial'] else 'promptFromDataset'
                    ],
                    'ppl': next(ppl_iterator),
                    'score': run['initial']['score']
                },
                'taken': [
                    {
                        'input_prompt_for_generation': taken['input_prompt_for_generation'],
                        'ppl': next(ppl_iterator),
                        'score': taken['score']
                    }
                    for taken in run['taken']
                ]
            }
            for run in data['runs']
        ]
    }
    if args.output == '':
        # Save data
        file_name = os.path.basename(args.data_path)
        dir_path = os.path.dirname(args.data_path)
        with open(os.path.join(dir_path, f'ppl_{file_name}'), 'w') as f:
            json.dump(results, f)
        logging.info(f"Results dumped at `{os.path.join(dir_path, f'ppl_{file_name}')}`")
    else:
        with open(args.output, 'w') as f:
            json.dump(results, f)
        logging.info(f"Results dumped at `{args.output}`")
    # Close script info
    logging.info("Script completed successfully")
    

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='evotox_prompts_perplexity',
        description='Script to compute the perplexity of a given LLM on a given set of prompts '
                    'generated with the evolutionary search'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--model',
        type=str,
        help="Model name on the HuggingFace hub (see https://huggingface.co/models)"
    )
    args_parser.add_argument(
        '--token',
        type=str,
        help="HuggingFace hub authentication token"
    )
    args_parser.add_argument(
        '--data_path',
        type=str,
        help="Path to the JSON file with the data generated by the evolutionary search"
    )
    args_parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="How many elements evaluate in a batch (speeds up the process, but may cause OOM Errors if too high)"
    )
    # output arguments
    args_parser.add_argument(
        '--output',
        type=str,
        default='',
        help="Path to the output file"
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
