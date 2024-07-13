# python script that takes as argument a path to load all the json files from
# then all the json are loaded into a list of Archive
# an Archive is composed by many elements, one of them, run, contains an element called taken, then for each run save the score in an array, divided by iterations
# finally plot the average score per iteration

import matplotlib.pyplot as plt
import json
import os
import click
import numpy as np
import pandas as pd
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolutionary import Archive, Run, InitialQuestion, Question
from chatAnalysis import get_score

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
    default="./results/oldButGold/100",
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
    default="tools/"
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
    default="svg",
    #type=click.Choice(["png", "pdf", "svg", "jpg", "jpeg", "tiff", "bmp", "gif", "eps"]),
    help="The extension of the output file",
)
def plot(input:str, silent, output, criteria, extension):
    archives = []
    df=pd.DataFrame()
    
    folders = [input]
    #folders = glob.glob(input + "/*")
    # remove the folder vicunaUC_vicunaUC from the list, if it exists
    if input + "/vicunaUC_vicunaUC" in folders:
        folders.remove(input + "/vicunaUC_vicunaUC")
    
    # remove everything that is not a folder
    #folders = [folder for folder in folders if os.path.isdir(folder)]

    for folder in folders:
        files = glob.glob(folder + "/*.json")
        for path in files:
            with open(path) as f:
                archive=Archive.from_dict(json.loads(f.read()))
                archives.append(average_score(archive))
    
    df=pd.concat(archives)
    '''
    #load all the json files in input
    files= glob.glob(input + "/*.json")
    for path in files:
        with open(path) as f:
            archives.append(Archive.from_dict(json.loads(f.read())))
            df=pd.concat([df, average_score(archives[-1])])
    '''
    # create subplots with two plots, one with the average score per iteration, and the other with the diff
    fig, ax = plt.subplots(2,1)
    # make the figure bigger
    fig.set_size_inches(5, 7)
    #ax[0].plot(df.mean(axis=0))
    #ax[0].set_title("Max, min and average score per iteration")
    #ax[0].plot(df.max(axis=0), linestyle="--")
    #ax[0].plot(df.min(axis=0), linestyle="--")
    #boxplot
    ax[0].boxplot(df, positions=np.arange(0, df.shape[1]), showfliers=False)
    ax[0].title.set_text("Score boxplot per iteration")
    #ax[0].legend(["Average", "Max", "Min"])
    ax[1].plot(df.mean(axis=0).diff())
    ax[1].set_title("Average score difference per iteration")

    mark=8
    # mark the 7th iteration with a vertical line and an horizontal line with the average score, printing the score rounded to 2 decimal places
    #ax[0].axvline(mark, color="red", linestyle="--")
    ax[1].axvline(mark, color="red", linestyle="--")
    ax[0].axhline(df.mean(axis=0)[mark], color="red", linestyle="--")
    ax[1].axhline(df.mean(axis=0).diff()[mark], color="red", linestyle="--")
    #ax[0].text(0, df.mean(axis=0)[mark], f"{df.mean(axis=0)[mark]:.2f}", ha="right")
    ax[1].text(mark, df.mean(axis=0).diff()[mark], f"{df.mean(axis=0).diff()[mark]:.2f}", ha="right")
    plt.tight_layout()

    name=os.path.basename(input)


    plt.savefig(os.path.join(output, "average_score"+name+"." + extension))
    plt.show()

if __name__ == "__main__":
    plot()

        