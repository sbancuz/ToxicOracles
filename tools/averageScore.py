# for each file .json in the folder given as a Click argument, compute the average score reached in the archive

import click
import json
import os
from evolutionary import Archive, Run, InitialQuestion, Question


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


def average_score(archive: Archive):
    # given an an archive, compute the average score reached by the last generation of each run
    scores = []
    for run in archive.runs:
        scores.append(run.taken[-1].score)
    return round(sum(scores) / len(scores), 4)
    



@click.command()
@click_option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False),
    help="Folder containing the archive files",
)
def printMaximum(input):
    # store the file name and the maximum score in a dictionary
    max_scores = {}

    for filename in os.listdir(input):
        if filename.endswith(".json"):
            with open(os.path.join(input, filename), "r") as file:
                archive = Archive.from_dict(json.load(file))
                max_scores[filename] = average_score(archive)
    # print the dictionary, ordered by the maximum score
    for filename, score in sorted(max_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{filename}: {score}")
                
                

if __name__ == "__main__":
    printMaximum()