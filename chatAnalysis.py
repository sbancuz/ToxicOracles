import matplotlib.pyplot as plt
import pandas as pd
import click
import matplotlib.ticker as mtick
import orjson
import os

from evolutionary import Archive, get_score


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


@click.command()
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=False, resolve_path=True, dir_okay=False),
    help="Path to source file, the output of evolutionary.py (output.json)",
    required=True,
    default="results/finalTests/vicuna_vicuna/max.json",
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
def plot(input, silent, output, criteria, extension):
    archive: Archive
    filename: str
    with open(input) as f:
        archive = Archive.from_dict(orjson.loads(f.read()))
        filename = os.path.splitext(os.path.basename(input))[0]

    scores = []
    categories = []
    timestamps = []

    # boolean parameter to check if there is the log (newer version of the output.json)

    logUpdate = archive.start_time_timestamp!=-1
    #logUpdate=True
    for run in archive.runs:
        iterationscores = []
        iterationcategories = []
        iterationscores.append(
            get_score(list(run.initial.criterion.values()), criteria)
        )
        for taken in run.taken:
            iterationscores.append(get_score(list(taken.criterion.values()), criteria))
            iterationcategories.append(taken.category)
            if logUpdate:
                timestamps.append([taken.delta_time_generation, taken.delta_time_response, taken.delta_time_evaluation])
        scores.append(iterationscores)
        categories.append(iterationcategories)

    df = pd.DataFrame(scores).transpose()
    df_cat = pd.DataFrame(categories).transpose()
    df_timestamps = pd.DataFrame(timestamps)

    fig, ax = plt.subplots(4, 3)
    # remove from source the path and the extension
    input = input.split("/")[-1].split(".")[0]

    # title of the plot
    fig.suptitle(
        "Analysis of " + input + " using " + criteria + "(criterions)", size=12
    )

    fig.tight_layout()

    ax[0, 0].set_title("Score", size=10)
    ax[0, 0].plot(df, alpha=0.3)
    # mean line with a red color and a label
    ax[0, 0].plot(df.mean(axis=1), color="red")

    ax[0, 1].set_title("Score difference over generations", size=10)
    ax[0, 1].plot(df.diff(), alpha=0.2)
    ax[0, 1].plot(df.diff().mean(axis=1), color="red")

    ax[0, 2].set_title("Score percent change over generations", size=10)
    ax[0, 2].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[0, 2].plot(df.pct_change(), alpha=0.2)
    ax[0, 2].plot(df.pct_change().mean(axis=1), color="red")

    # ax[3,0] shows the boxplot of the scores by each iteration
    ax[1, 0].set_title("Boxplot of scores by iteration", size=10)
    ax[1, 0].boxplot(df.T)

    ax[1, 1].set_title("Mean change for diff", size=10)
    ax[1, 1].plot(df.diff().mean(axis=1))

    ax[1, 2].set_title("Mean change for pct_change", size=10)
    ax[1, 2].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[1, 2].plot(df.pct_change().mean(axis=1))

    ax[2, 0].set_title("iterations with stable score", size=10)
    ax[2, 0].hist(df.diff().eq(0).sum(), alpha=0.5, color="blue", edgecolor="black")
    ax[2, 0].axvline(
        df.diff().eq(0).sum().mean(), color="red", linestyle="dashed", linewidth=1
    )

    ax[2, 1].set_title("Histogram of scores of the first iteration", size=10)
    ax[2, 1].hist(df.iloc[0], bins=30, alpha=0.5, color="blue", edgecolor="black")
    ax[2, 1].axvline(df.iloc[0].mean(), color="red", linestyle="dashed", linewidth=1)

    # ax[2,2] shows the histogram of scores of the last iteration
    ax[2, 2].set_title("Histogram of scores of the last iteration", size=10)
    ax[2, 2].hist(df.iloc[-1], bins=30, alpha=0.5, color="blue", edgecolor="black")
    ax[2, 2].axvline(df.iloc[-1].mean(), color="red", linestyle="dashed", linewidth=1)

    # ax[3,0] shows the boxplot of the gain by each category
    # boxplot of scores by category
    df_cat = df_cat.fillna("None")
    difference = df.diff().dropna()
    category_diff = {}

    for column in range(len(difference.columns)):
        for iteration in range(len(difference.index)):
            if difference.iloc[iteration, column] != 0:
                category = df_cat.iloc[iteration, column]
                if category not in category_diff:
                    category_diff[category] = []
                category_diff[category].append(difference.iloc[iteration, column])
    ax[3, 0].set_title("Boxplot of score gain by category", size=10)
    if (len(category_diff) != 0):
        ax[3, 0].boxplot(category_diff.values(), labels=category_diff.keys())
    ax[3, 0].set_xticklabels(ax[3, 0].get_xticklabels(), rotation=45, ha="right")

    if logUpdate:
        # pie chart of the time spent in each phase
        #ax[3, 1].set_title("Time spent in each phase", size=10)
        #print(df_timestamps.describe())
        #ax[3, 1].pie(df_timestamps.sum(), labels=["Generation", "Response", "Evaluation"], autopct="%1.1f%%")
        # boxplt of the time spent in each phase
        ax[3, 1].set_title("Time spent in each phase (s)", size=10)
        #ax[3, 1].boxplot(df_timestamps)
        # violin plot of the time spent in each phase
        if len(df_timestamps) > 0:
            ax[3, 1].violinplot(df_timestamps, showmeans=True)
            # set the labels of the x axis, set_ticks()
            ax[3, 1].set_xticks([1, 2, 3])
            ax[3, 1].set_xticklabels(["Generation", "Response", "Evaluation"])

    else:
        ax[3, 1].set_title("Category distribution per prompt", size=10)
        # do not show the legend
        df_cat.apply(pd.Series.value_counts).T.plot(
            kind="bar", stacked=True, ax=ax[3, 1], alpha=0.5, edgecolor="black", legend=False
        )

    
    ax[3, 2].set_title("Category distribution per iteration", size=10)
    if not df_cat.empty:
        df_cat.transpose().apply(pd.Series.value_counts).transpose().plot(
            kind="bar",
            stacked=True,
            ax=ax[3, 2],
            legend=False,
            alpha=0.5,
            edgecolor="black",
        )

    ax[3, 2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # set the label of 3,2 to be straight
    ax[3, 2].tick_params(axis="x", rotation=0)

    plt.subplots_adjust(right=0.9, wspace=0.175, bottom=0.11)
    fig.set_size_inches(13.5, 7.5)

    # save the plot
    plt.savefig(output + "/" + filename + "-" + criteria+"."+extension, dpi=300, format=extension)
    if not silent:
        plt.show()


if __name__ == "__main__":
    plot()
