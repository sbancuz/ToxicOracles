import os
import sys
import pandas as pd
import glob
import click
import orjson
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolutionary import Archive, get_score, Config
from sentenceAnalyser import distanceFromInitialPrompt, progressive

def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

def createFolderDf(input):
    '''
    Creates a dataframe with the following columns:
    - File, with the path of the file
    - SUT, with the name of the SUT
    - SG, with the name of the SG
    '''
    df = pd.DataFrame(columns=['File', 'SUT', 'SG'])
    folders= os.listdir(input)
    data=[]
    for folder in folders:
        # load json files
        files=glob.glob(input+folder+'/*.json')
        for file in files:
            archive=Archive.from_dict(orjson.loads(open(file, 'rb').read()))
            config=archive.config
            #print(config)
            data.append([file, config.system_under_test, config.prompt_generator])
    df=pd.DataFrame(data, columns=['File', 'SUT', 'SG'])
    #print(df)
    return df

def computeGeneralInitial(input, output, type, extension):
    df=createFolderDf(input)
    #obtain list of SUTs
    distances=[]

    files=df['File'].tolist()
    #plot the distance from the initial prompt of the files with the same SUT
    #print(filesSUT['File'].tolist())
    data=distanceFromInitialPrompt(files, 'violin', 'svg', '.', plot=False)
    # extract the similarities and the iterations
    for d in data:
        distances.append([d["cosine similarity"], d["iteration"], d["system_under_test"], d["prompt_generator"]])
    df=pd.DataFrame(distances, columns=['score', 'iteration', 'SUT', 'SG'])
   #print(df)

    #print(df)
    plot(df, 'violin', 'svg', '.', 'SUT', "Context Divergence from Initial Prompt")
    plot(df, 'violin', 'svg', '.', 'SG', "Context Divergence from Initial Prompt")
    plot(df, 'box', 'svg', '.', 'SUT', "Context Divergence from Initial Prompt")
    plot(df, 'box', 'svg', '.', 'SG', "Context Divergence from Initial Prompt")

    return df
        
def plot(data, kind='violin', format='svg', output='.', groupby='SUT', title=""):
    '''
    Plots the distance from the initial prompt of the files with the same SUT
    '''

    plt.figure(figsize=(20, 10))
    if kind=='violin':
        sns.violinplot(x='iteration', y='score', hue=groupby, data=data, palette='Set2')
    elif kind=='box':
        sns.boxplot(x='iteration', y='score', hue=groupby, data=data, palette='Set2')
    plt.title(title)

    plt.savefig(f'{output}/contextDivergence-{kind}-{groupby}.{format}', format=format)
    #plt.show()
    plt.close()

def computeGeneralProgressive(input):
    df=createFolderDf(input)
    #obtain list of SUTs
    distances=[]

    files=df['File'].tolist()
    #plot the distance from the initial prompt of the files with the same SUT
    #print(filesSUT['File'].tolist())
    data=progressive(files, 'violin', 'svg', '.', plot=False)
    # extract the similarities and the iterations
    for d in data:
        distances.append([d["cosine similarity"], d["iteration"], d["system_under_test"], d["prompt_generator"]])
    df=pd.DataFrame(distances, columns=['score', 'iteration', 'SUT', 'SG'])
   #print(df)
    return df
    #print(df)



@click.command()
@click_option("--input", 
              "-i", 
              default="./results/finalTests/", 
              help="Path to the folder with the json files.")
@click_option("--output",
                "-o",
                default="./results/finalTests/",
                help="Path to the folder where the output will be saved.")
@click_option("--type",
              "-t",
              type=click.Choice(["initial", "progressive", "all"]),
                default="all",
                help="Type of analysis to be performed.")
@click_option("--extension",
                "-e",
                default="svg",
                help="Format of the output file.")

def main(input, output, type, extension):
    if type=="initial" or type=="all":
        df=computeGeneralInitial(input, output, type, extension)
    elif type=="progressive" or type=="all":
        df=computeGeneralProgressive(input)
        print(df)
        plot(df, 'violin', 'svg', '.', 'SUT', "Progressive Context Divergence")
        plot(df, 'violin', 'svg', '.', 'SG', "Progressive Context Divergence")
        plot(df, 'box', 'svg', '.', 'SUT', "Progressive Context Divergence")
        plot(df, 'box', 'svg', '.', 'SG', "Progressive Context Divergence")

if __name__ == "__main__":
    main()