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
    folders = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]

    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            if file.startswith("ppl") and file.endswith(".json"):
                perplexityFiles.append(os.path.join(folder, file))
    print(perplexityFiles)
    return perplexityFiles

def sortValues(data):
    # create a dataframe with the perplexity values, the iteration and the model
    # data contains the list of files to read
    # return the dataframe
    values = []
    for file in data:
        with open(file, 'r') as f:
            data = json.load(f)
            model = data['model']
            for run in data['runs']:
                values.append([run['initial']['ppl'], 0, model])
                for i, taken in enumerate(run['taken']):
                    values.append([taken['ppl'], i+1, model])
    return pd.DataFrame(values, columns=['Perplexity', 'Iteration', 'Model'])

def plotPerplexity(data, output, folder, format='png'):
    # plot the data
    # save the plot in output
    print(data)
    sns.set_theme()
    sns.set_context("paper")
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Iteration', y='Perplexity', hue='Model')
    plt.savefig(f'{folder}{output}.{format}', dpi=300)

@click.command()
@click.option('--folder', '-f', help='Folder with the perplexity files', required=True, default='results/finalTests/')
@click.option('--output', '-o', help='Output file', required=True, default='perplexity')
@click.option('--format', '-t', help='Output format', required=False, default='png')

def main(folder, output, format):
    perplexityFiles = getPerplexityFiles(folder)
    data = sortValues(perplexityFiles)
    plotPerplexity(data, output, folder, format)

if __name__ == '__main__':
    main()