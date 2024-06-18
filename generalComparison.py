# given a folder in input, pick for each folder inside the given folder the archive that achieved the best average score at the end of the simulation.

import os
from typing import List

import click
import orjson
from evolutionary import Archive, Config
from comparison import get_files, get_score, get_legends, parse_config
import copy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

def pickBest(folder):
    # load all the json files in the folder
    files = get_files(folder)
    best=None
    bestAverage=None
    for file in files:
        with open(file) as f:
            data = Archive.from_dict(orjson.loads(f.read()))
            # compute the average score for the final iteration
            scoreSum=0
            for iteration in data.runs:
                scoreSum+=iteration.taken[-1].score
            average=scoreSum/len(data.runs)
            # if the current average is better than the current best, update the best
            if best==None or average>bestAverage:
                #deep copy the data
                best=copy.deepcopy(data)
                bestAverage=average
    return best

@click.command()

@click_option(
    "-i",
    "--input",
    default="results/finalTests/",
    help="The path to the results folder",
)
@click_option(
    "-o",
    "--output",
    default=".",
    help="The path to the output folder",
)
@click_option(
    "-e",
    "--extension",
    default="png",
    help="The extension of the output file",
)
@click_option(
    "-s",
    "--silent",
    is_flag=True,
    help="Do not show the plot",
)
def main(input, output, extension, silent):
    # for every folder in the input folder, compute the best archive
    folders = [f.path for f in os.scandir(input) if f.is_dir()]
    # data is an object that has the data of the run and a string for the legend
    data=[]
    criteria="max"

    for folder in folders:
        best=pickBest(folder)
        sut=best.config.system_under_test
        sg=best.config.prompt_generator

        data.append((best, "SUT: "+sut+" SG: "+sg+", "+parse_config(best.config)))
    
    legend = [d[1] for d in data]
    # plot the data
    fig, ax = plt.subplots(3, 2)
    fig.suptitle("Comparison of the different configurations", size=12)
    fig.tight_layout()
    # big window
    fig.set_size_inches(13, 7.5)
    # set the title of the plot

    ax[0, 0].set_title("Score", size=10)
    ax[0, 1].set_title("Score difference over the mean score", size=10)
    ax[1, 0].set_title("Score difference over generations", size=10)
    ax[1, 1].set_title("Score percentual change over generations", size=10)
    ax[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[2, 0].set_title(
        "Number of iterations in which the score has remained the same", size=10
    )
    ax[2, 1].set_title("Categories distribution", size=10)

    iterations = min([d[0].config.iterations for d in data])
    possible_categories = [d[0].config.categories for d in data][0]

    mean_iteration_score = None
    dataframes = []
    stationary_score = []
    categories_series = []

    for element in data:
        experiment=element[0]
        scores = []
        categories = []
        for run in experiment.runs:
            iterationscores = []
            iterationcategories = []
            iterationscores.append(
                get_score(list(run.initial.criterion.values()), criteria)
            )
            for i in range(1, iterations):
                taken = run.taken[i]
                iterationscores.append(
                    get_score(list(taken.criterion.values()), criteria)
                )
                iterationcategories.append(taken.category if taken.category != "" else "initial")

            scores.append(iterationscores)
            categories.append(iterationcategories)

        df = pd.DataFrame(scores).transpose()
        df_cat = pd.DataFrame(categories).transpose()

        dataframes.append(df)
        if mean_iteration_score is None:
            mean_iteration_score = df.mean(axis=1)
        else:
            mean_iteration_score += df.mean(axis=1)
        ax[0, 0].plot(df.mean(axis=1), alpha=1)
        ax[1, 0].plot(df.diff().mean(axis=1))
        ax[1, 1].plot(df.pct_change().mean(axis=1))
        stationary_score.append(df.diff().eq(0).sum().value_counts())
        categories_series.append(df_cat.apply(pd.Series.value_counts).transpose().sum())

    # compute the mean score
    mean_iteration_score = mean_iteration_score / len(data)
    for df in dataframes:
        ax[0, 1].plot(df.mean(axis=1) - mean_iteration_score)

    # plot the histogram of the number of iterations with the same score
    stationary_score = pd.DataFrame(stationary_score).T.sort_index()

    stationary_score = stationary_score.fillna(0)
    stationary_score.plot(
        kind="bar",
        stacked=False,
        ax=ax[2, 0],
        alpha=0.5,
        edgecolor="black",
        legend=False,
    )
    # ax[2, 0].hist(stationary_score, alpha=0.5, edgecolor="black")

    # plot the categories in a histogram for categories_series, labels are tilted for better readability
    categories_df = pd.DataFrame(categories_series)
    categories_df = categories_df.fillna(0)
    #categories_df = categories_df[possible_categories]

    categories_df.transpose().plot(
        kind="bar", stacked=False, ax=ax[2, 1], alpha=0.5, edgecolor="black"
    )
    ax[2, 1].set_xticklabels(categories_df.T.index, rotation=25, ha="right")

    ax[0, 1].legend(legend, loc="center left", bbox_to_anchor=(1, 0.5))
    # ax[2, 0].legend(legend)
    # legend is on the right, outside the plot
    ax[2, 1].legend(legend, loc="center left", bbox_to_anchor=(1, 0.5))
    # make the legend to be seen in the window
    plt.subplots_adjust(right=0.74, bottom=0.16, left=0.04, top=0.84)

    plt.savefig(output + f"/generalComparison-{criteria}.{extension}", dpi=300, format=extension)

    if not silent:
        plt.show()





if __name__ == "__main__":
    main()

