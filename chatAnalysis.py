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
def plot(input, silent, output, criteria):
    archive: Archive
    filename: str
    with open(input) as f:
        archive = Archive.from_dict(orjson.loads(f.read()))
        filename = os.path.splitext(os.path.basename(input))[0]

    scores = []
    categories = []
    for run in archive.runs:
        iterationscores = []
        iterationcategories = []
        iterationscores.append(
            get_score(list(run.initial.criterion.values()), criteria)
        )
        for taken in run.taken:
            iterationscores.append(get_score(list(taken.criterion.values()), criteria))
            iterationcategories.append(taken.category)
        scores.append(iterationscores)
        categories.append(iterationcategories)

    df = pd.DataFrame(scores).transpose()
    df_cat = pd.DataFrame(categories).transpose()

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

    ax[0, 2].set_title("Score percentual change over generations", size=10)
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
    ax[3, 0].boxplot(category_diff.values(), labels=category_diff.keys())
    ax[3, 0].set_xticklabels(ax[3, 0].get_xticklabels(), rotation=45, ha="right")

    ax[3, 1].set_title("Category distribution per iteration", size=10)
    df_cat.transpose().apply(pd.Series.value_counts).transpose().plot(
        kind="bar",
        stacked=True,
        ax=ax[3, 1],
        legend=False,
        alpha=0.5,
        edgecolor="black",
    )

    ax[3, 2].set_title("Category distribution per prompt", size=10)
    df_cat.apply(pd.Series.value_counts).T.plot(
        kind="bar", stacked=True, ax=ax[3, 2], alpha=0.5, edgecolor="black"
    )
    # make the legend outside the plot
    ax[3, 2].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(right=0.9, wspace=0.175, bottom=0.1)
    fig.set_size_inches(13.5, 7.5)

    # save the plot
    plt.savefig(output + "/" + filename + "-" + criteria + ".png", dpi=300)
    if not silent:
        plt.show()


if __name__ == "__main__":
    plot()
