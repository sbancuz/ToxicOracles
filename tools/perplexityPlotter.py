# creates a graph for each model regarding the perplexity obtained in each run


import click
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seaborn as sns
import matplotlib.pyplot as plt

def getPerplexityFiles(root):
    # for each folder in folder
    # get the perplexity files
    # return the list of files
    perplexityFiles = []
    # folders in the root folder
    folders = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))] #and folder != 'vicunaUC_vicunaUC']
    for folder in folders:
        # furthermore, search if there are folders named "perplexity" and add the files in those folders
        perplexityFolder = os.path.join(folder, 'perplexity')
        # if the folder exists
        if os.path.exists(perplexityFolder):
            # list the folders in perplexityFolder
            perpFolder= [os.path.join(perplexityFolder, folder) for folder in os.listdir(perplexityFolder) if os.path.isdir(os.path.join(perplexityFolder, folder))]
            for model in perpFolder:
                files = os.listdir(model)
                for file in files:
                    if file.startswith("ppl") and file.endswith(".json"):
                        perplexityFiles.append(os.path.join(model, file))
        else:
            # if the folder does not exist, list the files in the folder
            files = os.listdir(folder)
            for file in files:
                if file.startswith("ppl") and file.endswith(".json"):
                    perplexityFiles.append(os.path.join(folder, file))
    return perplexityFiles



def sortValues(data):
    # create a dataframe with the perplexity values, the iteration and the model
    # data contains the list of files to read
    # return the dataframe
    values = []
    iterations=11
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

                handle='Jailbreak '+model
                # case of baseline
                for run in data['runs']:
                    for i in range(iterations):
                        values.append([run['initial']['ppl'], i, sut, handle, model, run['initial']['score']])
    
    return pd.DataFrame(values, columns=['Perplexity', 'Iteration', 'SUT', 'Handle', 'Model', 'score'])

def plotPerplexity(data, output="perplexity", folder="./", format='png', hue="Handle"):
    # plot the data
    # save the plot in output
    modelNumber = len(data['SUT'].unique())
    # create a figure with as many subplots as models

    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))

    # for each sut plot the perplexity
    for i, sut in enumerate(data['SUT'].unique()):
        sns.lineplot(data=data[data['SUT'] == sut], x='Iteration', y='Perplexity', hue=hue, style=hue, markers=False, dashes=False, ax=axs[i], errorbar=None)
        axs[i].set_title(sut)

    #plt.show()
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


def boxplot(data, output="perplexity", folder="./", format='png'):
    modelNumber = len(data['SUT'].unique())
    # create a figure with as many subplots as models
    fig, axs = plt.subplots(modelNumber, 1, figsize=(12, 6*modelNumber))
    # plot the boxplots for each SUT
    for i, model in enumerate(data['SUT'].unique()):
        # create the dataframe, combining the baseline and the data
        dataPpl = data[data['SUT'] == model]
        sns.boxplot(data=dataPpl, x='Handle', y='Perplexity', ax=axs[i])
        axs[i].set_title(model)
        # set log scale on y
        axs[i].set_yscale('log')
        # tilt the x labels
        axs[i].tick_params(axis='x', rotation=10)
        # hide x title
        axs[i].set_xlabel('')


        # print the data relative to the highest perplexity, file and model
        print(f'{model} highest perplexity: {dataPpl["Perplexity"].max()}')
        #print(dataPpl[dataPpl['Perplexity'] == dataPpl['Perplexity'].max()])



    plt.savefig(f'{folder}{output}_box.{format}', dpi=300)



@click.command()
@click.option('--folder', '-f', help='Folder with the perplexity files', required=True, default='results/finalTests/')
@click.option('--output', '-o', help='Output file', required=True, default='perplexity')
@click.option('--format', '-t', help='Output format', required=False, default='png')

def main(folder, output, format):
    perplexityFiles = getPerplexityFiles(folder)
    data = sortValues(perplexityFiles)
    plotPerplexity(data, output, folder, format)
    #normalisedScorePlotter(data, baseline, output, folder, format)
    boxplot(data, output, folder, format)

if __name__ == '__main__':
    main()