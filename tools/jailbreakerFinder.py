# given a csv file with the following format:
# id,sut,pg,version,prompt,response,translation,max score,threat,insult,severe toxicity,toxicity,identity attack,profanity,#
# retrieve from a given json file the jailbroken prompt (a jailbroken prompt ends with the initial prompt)
# the final output is a csv file with the following format:
# id,sut,pg,version,prompt,response,translation,max score,threat,insult,severe toxicity,toxicity,identity attack,profanity,#



import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import click
from evolutionary import Archive, InitialQuestion, PerspectiveCriterion
import random

def find_jailbroken_prompt(archive: Archive, prompt, number: int=30):
    # extract from archive all the prompts that end with the given prompt
    jailbroken_prompts = []
    print(f"Searching for jailbroken prompts for prompt {prompt}")
    for run in archive.runs:
        #print(f"Checking run {run.initial.prompt_from_dataset}")
        # if the prompt is contained in the initial prompt
        if prompt in run.initial.prompt_from_dataset:
            jailbroken_prompts.append(run.initial)
    # shuffle the list
    random.shuffle(jailbroken_prompts)
    # return the first number of jailbroken prompts
    return jailbroken_prompts[:number]


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

@click.command()
@click_option('--input',
                '-i',
                type=click.Path(exists=True),
                default='results/finalTests/baselineJailbreak/JailbreakPrompts-llama3.json')
@click_option('--prompts',
                '-p',
                type=str,
                default='results/finalTests/baselineJailbreak/selectedPrompts.csv'
                )
@click_option('--output',
                '-o',
                type=click.Path(),
                default='results/finalTests/baselineJailbreak/jailbrokenPrompts.csv'
                )
@click_option("--number",
                "-n",
                type=int,
                default=30
                )
def run(input, prompts, output, number):
    # load the archive
    with open(input, 'r') as f:
        archive = Archive.from_dict(json.load(f))
    # load the prompts
    with open(prompts, 'r') as f:
        reader = csv.DictReader(f=f, fieldnames=["id", "sut", "pg", "version", "prompt", "response", "translation", "max score", "threat", "insult", "severe toxicity", "toxicity", "identity attack", "profanity", "#"], delimiter=',', quotechar='"')
        prompts = list(reader)
        # remove the header
        prompts = prompts[1:]
    # open the output file
    with open(output, 'w') as f:
        # add a field for the jailbroken prompt after the prompt
        reader.fieldnames.insert(reader.fieldnames.index('prompt')+1, 'jailbroken')
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        # for each prompt find the jailbroken prompt
        counter=1
        sut=archive.config.system_under_test
        pg=archive.config.prompt_generator


        for prompt in prompts:
            jailbroken_prompts = find_jailbroken_prompt(archive, prompt['prompt'], number)
            print(f"Found {len(jailbroken_prompts)} jailbroken prompts for prompt {prompt['prompt']}")
            for jailbroken in jailbroken_prompts:
                # replace new line characters
                jailbroken.prompt_from_dataset = jailbroken.prompt_from_dataset.replace('\n', ' ')

                writer.writerow({
                    'id': counter,
                    'sut': sut,
                    'pg': pg,
                    'version': "JailbreakPrompts",
                    'prompt': prompt['prompt'],
                    'jailbroken': jailbroken.prompt_from_dataset,
                    'response': jailbroken.response_from_sut,
                    'translation': "---",
                    'max score': jailbroken.score,
                    'threat': jailbroken.criterion.get('THREAT'),
                    'insult': jailbroken.criterion.get('INSULT'),
                    'severe toxicity': jailbroken.criterion.get('SEVERE_TOXICITY'),
                    'toxicity': jailbroken.criterion.get('TOXICITY'),
                    'identity attack': jailbroken.criterion.get('IDENTITY_ATTACK'),
                    'profanity': jailbroken.criterion.get('PROFANITY'),
                    '#': "---"
                })
                counter += 1
        # close the file
        f.close()

if __name__ == '__main__':
    run()


        
    