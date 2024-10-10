# creates a graph for each model regarding the perplexity obtained in each run


import click
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seaborn as sns
import matplotlib.pyplot as plt
from pplJsonFixer import getPerplexityFiles

def sortValues(data):
    # create a dataframe with the perplexity values, the iteration and the model
    # data contains the list of files to read
    # return the dataframe
    values = []
    for file in data:
        with open(file, 'r') as f:
            data = json.load(f)
            sut = data['sut']#+'/'+data['configuration']
            configuration = data['configuration']
            model = data['model']
            for run in data['runs']:
                values.append([run['initial']['ppl'], 0, sut, configuration, model, run['initial']['score']])
                for i, taken in enumerate(run['taken']):
                    values.append([taken['ppl'], i+1, sut, configuration, model, taken['score']])
    return pd.DataFrame(values, columns=['Perplexity', 'Iteration', 'SUT', 'Configuration', 'Model', 'score'])

def plotPerplexity(data, output, folder, format='png'):
    # plot the data
    # save the plot in output
    modelNumber = len(data['Model'].unique())
    # create a figure with as many subplots as models

    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))
    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("whitegrid")
    for i, model in enumerate(data['Model'].unique()):
        sns.lineplot(data=data[data['Model'] == model], x='Iteration', y='Perplexity', hue='SUT', style='Configuration', markers=True, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(model)

    plt.savefig(f'{folder}{output}.{format}', dpi=300)

def scorePlotter(data, output, folder, format='png'):
    modelNumber = len(data['Model'].unique())
    # create a figure with as many subplots as models
    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))

    # compute the ratio between the score and the perplexity
    try:
        data['Ratio'] = data['score']/data['Perplexity']
    except:
        print(data)

    for i, model in enumerate(data['Model'].unique()):
        sns.lineplot(data=data[data['Model'] == model], x='Iteration', y='Ratio', hue='SUT', style='Configuration', markers=True, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(model)
    
    plt.savefig(f'{folder}{output}_ratio.{format}', dpi=300)

def normalisedScorePlotter(data, output, folder, format='png'):
    modelNumber = len(data['Model'].unique())
    # create a figure with as many subplots as models
    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))
    # normalise the score for each model and sut combination
    data['NormalisedScore'] = data.groupby(['Model', 'SUT'])['score'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    # normalise the perplexity for each model and sut combination
    data['NormalisedPerplexity'] = data.groupby(['Model', 'SUT'])['Perplexity'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    # compute the ratio between the score and the perplexity
    data['Ratio'] = data['NormalisedScore']/data['NormalisedPerplexity']
    for i, model in enumerate(data['Model'].unique()):
        sns.lineplot(data=data[data['Model'] == model], x='Iteration', y='Ratio', hue='SUT', markers=True, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(model)


    plt.savefig(f'{folder}{output}_norm.{format}', dpi=300)



@click.command()
@click.option('--folder', '-f', help='Folder with the perplexity files', required=True, default='results/finalTests/')
@click.option('--output', '-o', help='Output file', required=True, default='perplexity')
@click.option('--format', '-t', help='Output format', required=False, default='png')

def main(folder, output, format):
    perplexityFiles = getPerplexityFiles(folder)
    data = sortValues(perplexityFiles)
    #plotPerplexity(data, output, folder, format)
    normalisedScorePlotter(data, output, folder, format)

if __name__ == '__main__':
    main()