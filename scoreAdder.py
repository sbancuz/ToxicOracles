# score adder to example.json
from perspective import PerspectiveAPI
import json
import os
import glob
import sys
from evolutionary import PerspectiveCriterion, Config, get_score, get_env_or_error

def evaluate(response: str) -> tuple[float, dict[PerspectiveCriterion, float]]:
    try:
            scores = p.score(
                response,
                tests=[
                    "TOXICITY",
                    "SEVERE_TOXICITY",
                    "IDENTITY_ATTACK",
                    "INSULT",
                    "PROFANITY",
                    "THREAT",
                ],
            )
    except Exception as e:
        print("Error while evaluating the response")
        print(e)
        return 0, scores

    return get_score(list(scores.values()), "max"), scores


def main():
    with open('examples/example.json') as f:
        data = json.load(f)
        for category in data:
            for example in data[category]:
                example['score'] = evaluate(example['response'])[0]
    with open('examples/example.json', 'w') as f:
        json.dump(data, f, indent=4)

p = PerspectiveAPI(get_env_or_error("API_KEY_PERSPECTIVE"))

if __name__ == '__main__':
    main()