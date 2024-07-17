import matplotlib.pyplot as plt
import pandas as pd
import click
import matplotlib.ticker as mtick

from evolutionary import Archive, Config, get_score


import orjson


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


def parse_config(config: Config) -> str:
    name = config.scoring_function + "_"

    if config.memory:
        name += f"mem({config.memorywindow})_"

    if config.forward_score:
        name += "fs_"

    if config.gaslight:
        name += "glit_"

    return name[:-1]


def get_files(path: str, includeBaseline=False) -> list[str]:
    '''
    path: the path to the folder containing the files
    includeBaseline: whether to include the baseline files or not
    Return a list of all the files in the path that have the .json extension and are not directories
    '''
    import os

    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith(".json") and (includeBaseline or "baseline" not in f)
            
    ]


def get_data(files: list[str]) -> list[Archive]:
    data = []
    for file in files:
        with open(file) as f:
            data.append(Archive.from_dict(orjson.loads(f.read())))

    return data


def get_legends(files: list[Archive]) -> list[str]:
    legends = []
    for file in files:
        legends.append(parse_config(file.config))

    return legends


@click.command()
@click_option(
    "-c",
    "--criteria",
    default="max",
    type=click.Choice(["max", "min", "avg", "median"]),
    help="The criteria to use for the score comparison",
)
@click_option(
    "-i",
    "--input",
    default="results/finalTests/llama3_llama3",
    help="The path to the results folder",
)
@click_option(
    "-s",
    "--silent",
    is_flag=True,
    help="Do not show the plot",
    default=False,
)
@click_option(
    "-o",
    "--output",
    default="results/finalTests/llama3_llama3/analysis/comparisons",
    help="The path to the results folder",
)
@click_option(
    "-e",
    "--extension",
    default="svg",
    type=click.Choice(["png", "pdf", "svg"]),
    help="The extension of the output file",
)
def plot(criteria: str, input: str, silent: bool, output: str, extension: str) -> None:
    '''
    Compare the different configurations of the runs in the input folder.
    criteria: the criteria to use for the score comparison
    input: the path to the results folder
    silent: do not show the plot
    output: the path to the results folder
    extension: the extension of the output file'''


    sources = get_files(input, includeBaseline=False)
    print(sources)

    data = get_data(sources)
    legend = get_legends(data)
    #legend is the name of the files
    #legend=[source.split("/")[-1] for source in sources]
    
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

    iterations = min([d.config.iterations for d in data])
    possible_categories = [d.config.categories for d in data][0]

    mean_iteration_score = None
    dataframes = []
    stationary_score = []
    categories_series = []

    for experiment in data:
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

    plt.savefig(output + f"/comparison-{criteria}.{extension}", dpi=300, format=extension)

    if not silent:
        plt.show()


if __name__ == "__main__":
    plot()
