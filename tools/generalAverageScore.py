# python script that takes as argument a path to load all the json files from
# then all the json are loaded into a list of Archive
# an Archive is composed by many elements, one of them, run, contains an element called taken, then for each run save the score in an array, divided by iterations
# finally plot the average score per iteration

from evolutionary import Archive, Run, InitialQuestion, Question
import matplotlib.pyplot as plt
import json
import os
import click
import numpy as np
from chatAnalysis import get_score
import pandas as pd
import glob

def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

def average_score(archive: Archive):
    scores = []
    for run in archive.runs:
        iterationscores = []
        iterationscores.append(
            get_score(list(run.initial.criterion.values()), "max")
        )
        for taken in run.taken:
            iterationscores.append(get_score(list(taken.criterion.values()), "max"))
        scores.append(iterationscores)

    df = pd.DataFrame(scores)
    return df

@click.command()
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=False),
    required=True,
    help="The path to the results folder",
)
@click_option(
    "--silent",
    is_flag=True,
    help="Do not show the plot, only save it",
    default=False,
)
@click_option(
    "-o",
    "--output",
    type=click.Path(exists=False, resolve_path=True, dir_okay=True),
    help="Path to save the output",
    required=False,
    default="."
)
@click_option(
    "-c",
    "--criteria",
    default="max",
    type=click.Choice(["max", "min", "avg", "median"]),
    help="The criteria to use for the score comparison",
)
@click_option(
    "-e",
    "--extension",
    default="png",
    type=click.Choice(["png", "pdf", "svg"]),
    help="The extension of the output file",
)
def plot(input:str, silent, output, criteria, extension):
    archives = []
    df=pd.DataFrame()
    
    #load all the json files in input
    files= glob.glob(input + "/*.json")
    for path in files:
        with open(path) as f:
            archives.append(Archive.from_dict(json.loads(f.read())))
            df=pd.concat([df, average_score(archives[-1])])
    # create subplots with two plots, one with the average score per iteration, and the other with the diff
    fig, ax = plt.subplots(1, 2)
    # make the figure bigger
    fig.set_size_inches(12, 5)
    ax[0].plot(df.mean(axis=0))
    ax[0].set_title("Average score per iteration")
    ax[1].plot(df.mean(axis=0).diff())
    ax[1].set_title("Average diff per iteration")

    # mark the 7th iteration with a vertical line and an horizontal line with the average score, printing the score rounded to 2 decimal places
    ax[0].axvline(7, color="red", linestyle="--")
    ax[1].axvline(7, color="red", linestyle="--")
    ax[0].axhline(df.mean(axis=0)[7], color="red", linestyle="--")
    ax[1].axhline(df.mean(axis=0).diff()[7], color="red", linestyle="--")
    ax[0].text(7, df.mean(axis=0)[7], f"{df.mean(axis=0)[7]:.2f}", ha="right")
    ax[1].text(7, df.mean(axis=0).diff()[7], f"{df.mean(axis=0).diff()[7]:.2f}", ha="right")
    plt.tight_layout()


    plt.savefig(os.path.join(output, "average_score." + extension))
    plt.show()

if __name__ == "__main__":
    plot()

        