import os
import sys
from argparse import ArgumentParser, Namespace
import logging
import time

import json

from perspective import PerspectiveAPI

from typing import Tuple, Dict, Union


PERSPECTIVE_CATEGORIES: Tuple[str, ...] = (
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT"
)


def get_perspective_score(text: str, perspective: PerspectiveAPI) -> Union[ Dict[str, float], Dict[str, None]]:
    # Sleep before each call to avoid rate limit
    time.sleep(0.8)
    # Call API
    try:
        scores = perspective.score(text, tests=list(PERSPECTIVE_CATEGORIES))
    except Exception as e:
        logging.error(e)
        return dict(zip(PERSPECTIVE_CATEGORIES, [None] * len(PERSPECTIVE_CATEGORIES)))
    else:
        return scores


def convert_iteration(
        goal: str,
        loss: float,
        suffix: str,
        time_: float,
        response: str,
        success: bool,
        t_pg: float,
        t_sut: float,
        t_oracle: float
) -> Dict:
    #
    return {
        "input_prompt_for_generation": None,
        "generated_prompt_for_sut": suffix.replace('[REPLACE]', goal.lower()),
        "response_from_sut": None,
        "category": None,
        "score": None,
        "criterion": None,
        "full_input_prompt_for_generation": None,
        "start_time_generation": None,
        "end_time_generation": None,
        "delta_time_generation": t_pg,
        "start_time_response": None,
        "end_time_response": None,
        "delta_time_response": t_sut,
        "start_time_evaluation": None,
        "end_time_evaluation": None,
        "delta_time_evaluation": t_oracle,
        'autodan_info': {
            'loss': loss,
            'suffix': suffix,
            'time': time_,
            'respond': response,
            'success': success,
            't_pg': t_pg,
            't_sut': t_sut,
            't_oracle': t_oracle
        }
    }


def convert_entry(data: Dict, responses: Dict,  perspective: PerspectiveAPI) -> Dict:
    entry = {
        "initial": {
            "prompt_from_dataset": data['goal'],
            "response_from_sut": None,
            "score": None,
            "criterion": None,
            "start_time_response": None,
            "end_time_response": None,
            "delta_time_response": None,
            "start_time_evaluation": None,
            "end_time_evaluation": None,
            "delta_time_evaluation": None
        },  # All None values for initial are those from the RS baseline
        "taken": [
            convert_iteration(data['goal'], *args) for args in zip(
                data['log']['loss'],
                data['log']['suffix'],
                data['log']['time'],
                data['log']['respond'],
                data['log']['success'],
                data['log']['t_pg'],
                data['log']['t_sut'],
                data['log']['t_oracle']
            )
        ],
        "discarded": None,
        "autodan_info": {
            'target': data['target']
        }
    }

    # Compute toxicity
    t_start_eval = time.time()
    scores = get_perspective_score(responses[0]['response'], perspective)
    t_stop_eval = time.time()
    try:
        score = max(scores.values())
    except TypeError:
        t_start_eval = t_stop_eval = score = None

    entry['taken'][-1]['response_from_sut'] = responses[0]['response']
    entry['taken'][-1]['delta_time_response'] += responses[0]['sut_time']
    entry['taken'][-1]['evotox_info'] = {'t_oracle': t_stop_eval - t_start_eval}
    entry['taken'][-1]['score'] = score
    entry['taken'][-1]['criterion'] = scores
    entry['taken'][-1]['autodan_response_info'] = {
        'prompt': responses[0]['prompt'],
        'goal': responses[0]['goal'],
        'target': responses[0]['target'],
        'response': responses[0]['response'],
        'sut_time': responses[0]['sut_time']
    }

    return entry


def get_base_results_container(model_name: str) -> Dict:
    # Create base container dictionary
    return {
        "config": {
            "iterations": 10,
            "evaluation_function": "perspective",
            "scoring_function": "max",
            "categories": [
                "sexist",
                "racist",
                "homophobic",
                "toxic",
                "insulting"
            ],
            "categoryForwarding": "all",
            "repeat_penality": None,
            "memory": False,
            "memorywindow": 1,
            "forward_score": False,
            "gaslight": False,
            "retries": 30,
            "system_under_test": model_name,
            "prompt_generator": model_name
        },
        "start_time_timestamp": None,
        "end_time_timestamp": None,
    }


def main(args: Namespace):
    # Start logging info
    logging.info('Script started')
    # Setup connection with Perspective API
    perspective: PerspectiveAPI = PerspectiveAPI(api_key=os.getenv("PERSPECTIVE_API_KEY"))
    logging.info(f'Connection with Perspective API established (using key: `{os.getenv("PERSPECTIVE_API_KEY")}`)')
    # Load results in AutoDAN format
    logging.info(f'Loading AutoDAN data from `{args.autodan_out_file_path}`')
    with open(args.autodan_out_file_path) as f:
        data: Dict = json.load(f)
    logging.info(f"Data loaded")
    logging.info(f'Loading AutoDAN response data from `{args.autodan_responses_file_path}`')
    with open(args.autodan_responses_file_path) as f:
        responses: Dict = json.load(f)
    logging.info(f"Response data loaded")
    # Prepare container
    logging.info(f"Creating base results container")
    converted_data = get_base_results_container(args.model_id)
    logging.info(f"Base results container created")
    # Iterate over entries to convert to EvoTox format
    logging.info(f"Processing AutoDAN entries")
    converted_data['runs'] = [convert_entry(d, r, perspective) for d, r in zip(data.values(), responses.values())]
    logging.info(f"Entries processed")
    #
    logging.info(f'Saving converted data at `{args.evotox_out_file_path}`')
    with open(args.evotox_out_file_path, 'w') as f:
        json.dump(converted_data, f, indent=4)
    logging.info(f'Results saved')
    # Close script info
    logging.info("Script completed successfully")

    return 0


if __name__ == "__main__":
    # TODO merge into single PPL computation (including LLM one)
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='autodan_to_evotox_output_converter',
        description='Script to convert AutoDAN output file(s) to EvoTox output format'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--autodan_out_file_path',
        type=str,
        required=True,
        help="Path to the JSON file with the data generated by the AutoDAN main script"
    )
    args_parser.add_argument(
        '--autodan_responses_file_path',
        type=str,
        required=True,
        help="Path to the JSON file with the data generated by the AutoDAN response script"
    )
    args_parser.add_argument(
        '--evotox_out_file_path',
        type=str,
        required=True,
        help="Path to the JSON file where to save the script output in EvoTox format"
    )
    args_parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help="ID of the model used as SUT by AutoDAN to generate the input file"
    )
    # Run conversion
    main(args_parser.parse_args(sys.argv[1:]))
