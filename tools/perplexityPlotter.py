# creates a graph for each model regarding the perplexity obtained in each run


import click
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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



def sortValues(data: list, deleteOutliers: int=None):
    '''
    This function reads the list perplexity files and creates a dataframe with the perplexity values
    data: list of files to read
    deleteOutliers: if not None, delete the outliers with 
    '''
    # create a dataframe with the perplexity values, the iteration and the model
    # data contains the list of files to read
    # return the dataframe
    values = []
    iterations=11
    for file in tqdm(data):
        with open(file, 'r') as f:
            data = json.load(f)
            sut = data['config']['systemUnderTest']#+'/'+data['configuration']
            
            handle = data['handle']
            try:
                model = data['model']
            except:
                # name of the model is not present in the file
                model=str(data['n-gram']['order'])+"-gram_"+data['n-gram']['training_corpus']+'_arpa'

            if data['runs'][0]['taken'].__len__() > 1:
                for run in data['runs']:
                    values.append([run['initial']['ppl'], 0, sut, handle, model, run['initial']['score'], run['initial']['prompt_from_dataset'], file])
                    for i, taken in enumerate(run['taken']):
                        #print(taken)
                        values.append([taken['ppl'], i+1, sut, handle, model, taken['score'], taken['input_prompt_for_generation'], file])
            else:

                #handle='Jailbreak '+model
                # case of baseline
                for run in data['runs']:
                    for i in range(iterations):
                        values.append([run['initial']['ppl'], i, sut, handle, model, run['initial']['score'], run['initial']['prompt_from_dataset'], file])
            if deleteOutliers:
                # delete the outliers, i.e. the values that are more than 10 standard deviations from the mean

                # non il massimo dell'efficienza, ma funziona
                df = pd.DataFrame(values, columns=['Perplexity', 'Iteration', 'SUT', 'Handle', 'Model', 'score', 'prompt', 'file'])
                df = df[np.abs(df['Perplexity']-df['Perplexity'].mean()) <= (deleteOutliers*df['Perplexity'].std())]
                values = df.values.tolist()


    return pd.DataFrame(values, columns=['Perplexity', 'Iteration', 'SUT', 'Handle', 'Model', 'score', 'prompt', 'file'])

def plotPerplexity(data, output="perplexity", folder="./", format='png', hue="Handle", pplModels=None, sut=None):
    '''
    This function plots the perplexity for each model and sut combination
    data: dataframe with the perplexity values, in the format returned by sortValues
    output: name of the output file
    folder: folder where to save the output
    format: format of the output file
    hue: column to use for the hue, default is "Handle"
    pplModels: list of models to plot, default is None and all the models are plotted
    sut: list of suts to plot, default is None and all the suts are plotted
    '''
    # plot the data
    # save the plot in output


    # in the model column replace "eleutherai" with "EleutherAI", my mistake
    data['Model'] = data['Model'].replace('eleutherai/pythia-12b', 'EleutherAI/pythia-12b')

    # replace dots with underscores
    data['Model'] = data['Model'].str.replace('.', '_')

    # replace - in the handle with _
    data['Handle'] = data['Handle'].str.replace('-', '_')

    # lower case for all handles
    data['Handle'] = data['Handle'].str.lower()

    # remove the part after _ in the handles that start with jailbreakprompts
    data['Handle'] = data['Handle'].apply(
    lambda x: x.split('_')[0] if x.startswith('jailbreakprompts') else x
    )


    # create a figure with as many subplots as models
    if sut:
        modelNumber = len(sut)
    else:
        modelNumber = len(data['SUT'].unique())

    if pplModels:
        pplModel = sorted(pplModels)
    else: 
        pplModel=data['Model'].unique()
        pplModel = sorted(pplModel)

    fig, axs = plt.subplots(len(pplModel), modelNumber, figsize=(5*modelNumber, 4*len(pplModel)), sharex=True, sharey="row")

    # # for each sut plot the perplexity
    # for i, sut in enumerate(data['SUT'].unique()):
    #     sns.lineplot(data=data[data['SUT'] == sut], x='Iteration', y='Perplexity', hue=data[hue].apply(tuple, axis=1), style=data[hue].apply(tuple, axis=1), markers=False, dashes=False, ax=axs[i], errorbar=None)
    #     #sns.lineplot(data=data[data['SUT'] == sut], x='Iteration', y='Perplexity', col="Model", hue=data[hue].apply(tuple, axis=1), style=data[hue].apply(tuple, axis=1), markers=False, dashes=False, ax=axs[i], errorbar=None)
    #     axs[i].set_title(sut)

    #order the pplModel
    



    # for each pplModel
    for i, model in enumerate(pplModel):
        #print(pplModel)
        for j, sut in enumerate(data['SUT'].unique()):
            plotData = data[(data['SUT'] == sut) & (data['Model'] == model)]
            #print(model)
            if not plotData.empty:
                #print(plotData)
                # handel the case of only one subplot
                if len(pplModel) == 1:
                    sns.lineplot(data=plotData, x='Iteration', y='Perplexity', hue=hue, style=hue, markers=False, dashes=False, ax=axs[j], errorbar=None)
                    axs[j].set_title(sut+" by "+model)
                    #axs[j].set_yscale('log')
                    axs[j].set_xlabel('')
                else:
                    sns.lineplot(data=plotData, x='Iteration', y='Perplexity', hue=hue, style=hue, markers=False, dashes=False, ax=axs[i,j], errorbar=None)
                    axs[i,j].set_title(sut+" by "+model)
                    #axs[i,j].set_yscale('log')
                    axs[i,j].set_xlabel('')
            else:
                axs[i,j].axis('off')
            
            # print the maximum perplexity for each model and the relative data
            print(f'{model} highest perplexity: {plotData["Perplexity"].max()}')
            print(plotData[plotData['Perplexity'] == plotData['Perplexity'].max()])
            # print the prompt with the highest perplexity
            print(f'{model} highest perplexity prompt: {plotData[plotData["Perplexity"] == plotData["Perplexity"].max()]}')
            
            
    
    #plt.show()
    plt.tight_layout()
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
    #boxplot(data, output, folder, format)

if __name__ == '__main__':
    main()