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
    baselineValues = []
    for file in data:
        with open(file, 'r') as f:
            data = json.load(f)
            sut = data['config']['systemUnderTest']#+'/'+data['configuration']
            
            handle = data['handle']
            model = data['model']
            if data['runs'][0]['taken'].__len__() > 1:
                for run in data['runs']:
                    values.append([run['initial']['ppl'], 0, sut, handle, model, run['initial']['score']])
                    for i, taken in enumerate(run['taken']):
                        values.append([taken['ppl'], i+1, sut, handle, model, taken['score']])
            else:
                handle='ppl_'+model
                # case of baseline
                for run in data['runs']:
                    baselineValues.append([run['initial']['ppl'], 0, sut, handle, model, run['initial']['score']])
    
    return pd.DataFrame(values, columns=['Perplexity', 'Iteration', 'SUT', 'Handle', 'Model', 'score']), pd.DataFrame(baselineValues, columns=['Perplexity', 'Iteration', 'SUT', 'Handle', 'Model', 'score'])

def plotPerplexity(data, baseline, output, folder, format='png'):
    # plot the data
    # save the plot in output
    modelNumber = len(data['SUT'].unique())
    iterations=data['Iteration'].max()+1

    # extend the baseline dataframe to have the same number of iterations as the data
    for i in range(iterations):
        newBaseline = baseline.copy()
        newBaseline['Iteration'] = i
        baseline = pd.concat([baseline, newBaseline])
    
    # merge the dataframes
    data = pd.concat([data, baseline])


    # create a figure with as many subplots as models

    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))

    # for each sut plot the perplexity
    for i, sut in enumerate(data['SUT'].unique()):
        sns.lineplot(data=data[data['SUT'] == sut], x='Iteration', y='Perplexity', hue='Handle', style='Handle', markers=True, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(sut)

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
        sns.lineplot(data=data[data['Model'] == model], x='Iteration', y='Ratio', hue='SUT', style='Handle', markers=True, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(model)
    
    plt.savefig(f'{folder}{output}_ratio.{format}', dpi=300)

def normalisedScorePlotter(data, baseline, output, folder, format='png'):
    modelNumber = len(data['Model'].unique())
    # create a figure with as many subplots as models
    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))
    # normalise the score for each model and sut combination
    data['NormalisedScore'] = data.groupby(['Model', 'SUT'])['score'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    baseline['NormalisedScore'] = baseline.groupby(['Model', 'SUT'])['score'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    # normalise the perplexity for each model and sut combination
    data['NormalisedPerplexity'] = data.groupby(['Model', 'SUT'])['Perplexity'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    baseline['NormalisedPerplexity'] = baseline.groupby(['Model', 'SUT'])['Perplexity'].transform(lambda x: (x-x.min())/(x.max()-x.min()))
    # compute the ratio between the score and the perplexity
    data['Ratio'] = data['NormalisedScore']/data['NormalisedPerplexity']
    baseline['Ratio'] = baseline['NormalisedScore']/baseline['NormalisedPerplexity']
    for i, model in enumerate(data['Model'].unique()):
        sns.lineplot(data=data[data['Model'] == model], x='Iteration', y='Ratio', hue='SUT', markers=True, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(model)
        # plot an horizontal line for the baseline
        baselinePpl = baseline[baseline['Model'] == model]
        axs[i].axhline(y=baselinePpl['Ratio'].mean(), color='r', linestyle='--')



    plt.savefig(f'{folder}{output}_norm.{format}', dpi=300)


def boxplot(data, baseline, output, folder, format='png'):
    modelNumber = len(data['SUT'].unique())
    # create a figure with as many subplots as models
    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))
    # plot the boxplots for each SUT
    for i, model in enumerate(data['SUT'].unique()):
        # create the dataframe, combining the baseline and the data
        baselinePpl = baseline[baseline['SUT'] == model]
        baselinePpl['Handle'] = 'JailbreakPrompt '
        dataPpl = data[data['SUT'] == model]
        dataPpl = pd.concat([dataPpl, baselinePpl])
        sns.boxplot(data=dataPpl, x='Handle', y='Perplexity', ax=axs[i])
        axs[i].set_title(model)
        # set log scale on y
        axs[i].set_yscale('log')

        # print the data relative to the highest perplexity, file and model
        print(f'{model} highest perplexity: {dataPpl["Perplexity"].max()}')
        print(dataPpl[dataPpl['Perplexity'] == dataPpl['Perplexity'].max()])



    plt.savefig(f'{folder}{output}_box.{format}', dpi=300)



@click.command()
@click.option('--folder', '-f', help='Folder with the perplexity files', required=True, default='results/finalTests/')
@click.option('--output', '-o', help='Output file', required=True, default='perplexity')
@click.option('--format', '-t', help='Output format', required=False, default='png')

def main(folder, output, format):
    perplexityFiles = getPerplexityFiles(folder)
    data, baseline = sortValues(perplexityFiles)
    plotPerplexity(data, baseline, output, folder, format)
    #normalisedScorePlotter(data, baseline, output, folder, format)
    boxplot(data, baseline, output, folder, format)

if __name__ == '__main__':
    main()