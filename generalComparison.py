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
import seaborn as sns
import math


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

def categories_distribution(categories_df, output, extension, verbose, legend):
    # plot the categories distribution in a separate plot
    #plt.figure()
    # set size
    plt.figure(figsize=(25, 8))

    plt.title("Categories distribution")
    categories_df.transpose().plot(kind="bar", stacked=False, alpha=0.5, edgecolor="black")
    plt.xticks(rotation=25, ha="right")
    plt.legend(legend, loc="center left", bbox_to_anchor=(1, 0.5))
    # set the right padding to show the legend
    plt.subplots_adjust(right=0.7, left=0.04, top=0.84)
    plt.savefig(output + f"/categoriesDistribution.{extension}", dpi=300, format=extension)
    if verbose:
        plt.show()
    plt.close()


def all(input, output, extension, verbose, criteria, type):
    '''
    This function is used to compare the different configurations of the experiments, considering the best configuration for each experiment. (pair of SUT and SG)
    input: the path to the folder containing the results of the experiments (each experiment is in a different folder)
    output: the path to the output folder
    extension: the extension of the output file
    verbose: whether to show the plot or not
    '''

    data=[]
    for folder in input:
        #print("Processing folder: "+folder)
        best=pickBest(folder)
        if best==None:
            #print("No files found in folder: "+folder)
            continue
        else:
            sut=best.config.system_under_test
            sg=best.config.prompt_generator

            data.append((best, "SUT: "+sut+" SG: "+sg+", "+parse_config(best.config)))
    
    legend = [d[1] for d in data]
    # plot the data
    fig, ax = plt.subplots(3, 2)
    fig.suptitle("Comparison of the different configurations", size=12)
    fig.tight_layout()
    # big window
    fig.set_size_inches(15, 8)
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
    plt.subplots_adjust(right=0.75, bottom=0.16, left=0.04, top=0.84)

    plt.savefig(output + f"/generalComparison-{criteria}.{extension}", dpi=300, format=extension)

    if verbose:
        plt.show()
    plt.close()
    categories_distribution(categories_df, output, extension, verbose, legend)

def grouped(input, output, extension, verbose, groupby, criteria, type):
    data=pd.DataFrame()
    for folder in input:
        # compute the mean score in each iteration, for each run considering all the files
        configurationData=pd.DataFrame()
        for file in get_files(folder):
            with open(file) as f:
                archive=Archive.from_dict(orjson.loads(f.read()))
                for run in archive.runs:
                    iterationscores=[get_score(list(run.initial.criterion.values()), criteria)]
                    for i in range(archive.config.iterations):
                        taken=run.taken[i]
                        iterationscores.append(get_score(list(taken.criterion.values()), criteria))
                    configurationData=pd.concat([configurationData, pd.DataFrame(iterationscores).transpose()])
        # compute the mean score for each iteration
       # meanScore=configurationData.mean()
        # add the sut and sg to the data
        configurationData["system_under_test"]=archive.config.system_under_test
        configurationData["prompt_generator"]=archive.config.prompt_generator
        data=pd.concat([data, pd.DataFrame(configurationData)])

    
    if groupby=="system_under_test":
        data=data.drop("prompt_generator", axis=1)
    elif groupby=="prompt_generator":
        data=data.drop("system_under_test", axis=1)
    
    

    if type=="line":
        plt.figure()
        # compute the mean score for each group
        data=data.groupby([groupby]).mean()
        data=data.transpose()
        # plot the data on a single figure
        
        data.plot()
        plt.legend(title=groupby)
        
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        title=f"Comparison of the different {groupby.replace('_', ' ')}"
        plt.title(title)
    elif type=="violin" or type=="boxplot":

        columns = data.columns[:-1]
    
        # Calculate the number of columns for 3 rows layout
        n_cols = math.ceil(len(columns) / 3)
        
        # Set up the matplotlib figure
        fig, axes = plt.subplots(3, n_cols, figsize=(20, 12), sharey=True)

        title=f"Comparison of the different {groupby.replace('_', ' ')}"
        fig.suptitle(title)
        
        # Flatten the axes array for easy indexing
        axes = axes.flatten()
        
        # Generate a violin plot for each column
        for i, column in enumerate(columns):
            if type == "boxplot":
                sns.boxplot(x=groupby, y=column, data=data, ax=axes[i], palette='Set2', hue=groupby)
            else:
                sns.violinplot(x=groupby, y=column, data=data, ax=axes[i], palette='Set2', hue=groupby)
            axes[i].set_title(f'Iteration {column}')
            axes[i].set_ylabel('Score')
            # Remove the x-axis label
            axes[i].set_xlabel('')
            # Rotate the x-axis labels for better readability
            axes[i].set_xticks(axes[i].get_xticks())
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        # Adjust layout
        plt.tight_layout()

    else:
        raise ValueError("Invalid type parameter")

    
    plt.savefig(output + f"/groupedComparison-{type}-{criteria}-{groupby}.{extension}", dpi=300, format=extension)
    if verbose:
        plt.show()
    plt.close()
    
def load_data(input, criteria):
    fileData=[]

    for folder in input:
        for file in get_files(folder):
            with open(file) as f:
                archive=Archive.from_dict(orjson.loads(f.read()))
                for run in archive.runs:
                    for i in range(archive.config.iterations):
                        fileData.append([i, get_score(list(run.taken[i].criterion.values()), criteria), archive.config.system_under_test, archive.config.prompt_generator, run.taken[i].delta_time_evaluation, run.taken[i].delta_time_generation, run.taken[i].delta_time_response])
    data=pd.DataFrame(fileData, columns=["iteration", "score", "system_under_test", "prompt_generator", "delta_time_evaluation", "delta_time_generation", "delta_time_response"])
    return data


def groupedMerged(input, output, extension, verbose, groupby, criteria, type, analysis):
    '''
    This function is used to plot the results of the experiments grouped by the given criteria (sut or sg) and merged in a single plot.
    input: the path to the folder containing the results of the experiments (each experiment is in a sub-folder)
    output: the path to the output folder
    extension: the extension of the output file
    verbose: whether to show the plot or not
    groupby: the criteria to use for grouping the score (sut or sg)
    criteria: the criteria to use for selecting the best score (max, min, avg, median)
    type: the type of plot (line, boxplot, violin)
    analysis: the type of analysis to perform (score, time)
    '''
    data = load_data(input, criteria)
    data = data.sort_values(by=groupby)
    plt.figure(figsize=(13, 6))
    if analysis=="score":
        if type=="boxplot":
            # Create the boxplot
            sns.boxplot(x='iteration', y='score', hue=groupby, data=data, palette='Set2')
        else:
            sns.violinplot(x='iteration', y='score', hue=groupby, data=data, palette='Set2')

        # Set plot title and labels
        plt.title("Score by "+groupby.replace("_", " ")+" and Iteration")
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend(title='System')
    else:
        # drop data with time equal to 0 or -1 or NaN
        data = data[(data["delta_time_evaluation"] > 0) & (data["delta_time_generation"] > 0) & (data["delta_time_response"] > 0)]
        # drop the iteration and score columns
        data = data.drop(["iteration", "score"], axis=1)
        #print(data)

        # create the columns time and category, being generation, response or evaluation
        data = pd.melt(data, 
                    id_vars=['system_under_test', 'prompt_generator'], 
                    value_vars=['delta_time_evaluation', 'delta_time_generation', 'delta_time_response'],
                    var_name='type', 
                    value_name='time')        
        #print(data)
        # sort the data by the groupby column
        data = data.sort_values(by=groupby)
        
        if type=="boxplot":
            sns.boxplot(x="type", y="time", hue=groupby, data=data, palette='Set2')
        else:
            sns.violinplot(x="type", y="time", hue=groupby, data=data, palette='Set2')
        # set the plot title and labels
        plt.title("Time by "+groupby.replace("_", " ")+" and type")
        plt.xlabel('Type')
        plt.ylabel('Time (s)')
        plt.legend(title='System')
    
    # Save the plot
    plt.savefig(output + f"/groupedMergedComparison-{analysis}-{type}-{criteria}-{groupby}.{extension}", dpi=300, format=extension)
    if verbose:
        # Display the plot
        plt.show()
                        

@click.command()

@click_option(
    "-i",
    "--input",
    default="results/finalTests",
    help="The path to the results folder",
)
@click_option(
    "-o",
    "--output",
    default="results/finalTests",
    help="The path to the output folder",
)
@click_option(
    "-e",
    "--extension",
    default="svg",
    help="The extension of the output file",
)
@click_option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Do not show the plot",
)
@click_option(
    "-g",
    "--groupby",
    default="all",
    help="The criteria to use for grouping the score (no, sut, sg, all)",
    type=click.Choice(["no", "sut", "sg", "all"]),
)

@click_option(
    "-c",
    "--criteria",
    default="max",
    help="The criteria to use for selecting the best score (max, min)",
    type=click.Choice(["max", "min", "avg", "median"]),
)

@click_option(
    "-t",
    "--type",
    default="line",
    help="The type of plot (line, boxplot, violinplot)",
    type=click.Choice(["line", "boxplot", "violin"]),
)

@click.option(
    "-m",
    "--merged",
    is_flag=True,
    help="When plotting the grouped results (by sut or sg), merge the results in a single plot",
)

@click.option(
    "-a",
    "--analysis",
    default="score",
    help="The type of analysis to perform (score, time) on the data",
    type=click.Choice(["score", "time"]),
)
def main(input, output, extension, verbose, groupby, criteria, type, merged, analysis):
    # for every folder in the input folder, compute the best archive
    folders = [f.path for f in os.scandir(input) if f.is_dir()] #and f.name != "vicunaUC_vicunaUC"]


    if groupby == "no" or groupby == "all":
        all(folders, output, extension, verbose, criteria, type)
    if groupby == "sut" or groupby == "all" or groupby == "sg":
        if not merged:
            if groupby == "sut" or groupby == "all":
                grouped(folders, output, extension, verbose, "system_under_test", criteria, type)
            if groupby == "sg" or groupby == "all":
                grouped(folders, output, extension, verbose, "prompt_generator", criteria, type)
        else:
            if groupby == "sut" or groupby == "all":
                groupedMerged(folders, output, extension, verbose, "system_under_test", criteria, type, analysis)
            if groupby == "sg" or groupby == "all":
                groupedMerged(folders, output, extension, verbose, "prompt_generator", criteria, type, analysis)
    

if __name__ == "__main__":
    main()

