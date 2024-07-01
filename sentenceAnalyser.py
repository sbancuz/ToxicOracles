from sentence_transformers import SentenceTransformer, util
from evolutionary import Archive
import orjson
import click
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torch.nn import functional as F


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


def similarityBetween(model, sentence1, sentence2):
    '''this function computes the cosine similarity between two sentences
    model: the model used to compute the embeddings
    sentence1: the first sentence
    sentence2: the second sentence
    return: the cosine similarity between the two sentences  
    '''


    #Compute embedding for both lists
    embedding_1= model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)
    #Compute cosine-similarits
    cosine_scores = float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0])
    return cosine_scores


def createPlot(instances, output, type, filename, mode, x, y, extension):
    '''this function creates a boxplot or violin plot for a file, showing the cosine similarity between sentences at each iteration
    instances: the instances to plot
    output: the path to save the plot
    type: the type of plot to generate
    filename: the file to analyse, used for the title
    mode: the mode of the analysis
    x: the x label of the plot
    y: the y label of the plot
    '''

    # file name without path and extension and removing progressive from the name
    filename = os.path.basename(filename)
    # remove the extension
    filename = os.path.splitext(filename)[0]
    # remove the mode from the name
    filename = filename.replace(mode, "")



    df = pd.DataFrame(instances)
    #violin plot, divided by file name
    plt.figure(figsize=(20, 10))
    if type == "boxplot":
        ax = sns.boxplot(x=x, y=y, data=df)
    else:
        ax = sns.violinplot(x=x, y=y, data=df)
    
    ax.set_title(f"{y} of {filename} ({mode}), by {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # save figure in file folder, with the type of plot
    plt.savefig(f"{output}/{filename}_{mode}_{type}.{extension}", dpi=300, format=extension)
    plt.close()

#### PROGRESSIVE SIMILARITY ####

def analyseProgressiveSimilarity(file):
    '''this function analyses the progressive similarity between prompts in the file, starting from the prompt from the dataset
    Note: the similarity is computed if and only if the prompt is different from the previous one

    file: the file containing the archive
    return: a list of dictionaries, each dictionary contains the starting prompt, the generated prompt, the cosine similarity between the two prompts, the configuration of the archive, the file name and the iteration

    '''
    archive: Archive
    instances=[]
    with open(file) as f:
        archive = Archive.from_dict(orjson.loads(f.read()))

    for run in tqdm(archive.runs):
        for i in range(0, len(run.taken)):
            filename = os.path.basename(file)
            # remove the extension
            filename = os.path.splitext(filename)[0]
            if i == 0:
                if (
                    run.initial.prompt_from_dataset.lower().strip()
                    != run.taken[i].generated_prompt_for_sut.lower().strip()
                ):
                    instances.append({
                        "starting": run.initial.prompt_from_dataset, 
                        "generated": run.taken[i].generated_prompt_for_sut, 
                        "cosine similarity": similarityBetween(model, 
                                                                   run.initial.prompt_from_dataset, 
                                                                   run.taken[i].generated_prompt_for_sut), 
                        "config": archive.config, 
                        "file name": filename, 
                        "iteration": i
                        })
            else:
                if run.taken[i - 1].generated_prompt_for_sut.lower().strip() != run.taken[i].generated_prompt_for_sut.lower().strip():
                    #instances.append({"starting": run.taken[i - 1].generated_prompt_for_sut, "generated": run.taken[i].generated_prompt_for_sut, "cosine similarity": float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]), "config": archive.config, "fileName": filename})
                    instances.append({
                                    "starting": run.taken[i - 1].generated_prompt_for_sut, 
                                    "generated": run.taken[i].generated_prompt_for_sut, 
                                    "cosine similarity": similarityBetween(model, 
                                                                        run.taken[i - 1].generated_prompt_for_sut, 
                                                                        run.taken[i].generated_prompt_for_sut), 
                                    "config": archive.config, 
                                    "file name": filename,
                                    "iteration": i                                    
                                    })
    return instances


def progressive(input, type, extension, output, savejson=False):
    ''' the progressive analysis is aimed at analysing the similarity between consecutive prompts, starting from the prompt from the dataset
    it saves the results in a file named as the file analysed, with the suffix _progressive.json and creates a plot for each file, saved in the same folder as well.
    input: the path to the folder containing the files to analyse
    type: the type of plot to generate
    '''
    instances=[]

    # get all files in the folder
    files = glob.glob(input+"/*.json")

    # create the folder progressive in the input folder, if it does not exist
    path=os.path.dirname(input)
    if not os.path.exists(output):
        os.makedirs(output)

    # iterate over files
    for file in files:
        print(f"analysing {file}")
        progressive_instances = analyseProgressiveSimilarity(file)
        if savejson:
            # save the instances in a file
            with open(f"{output}/{os.path.basename(file).replace('.json', '')}_progressive.json", "w") as f:
                f.write(orjson.dumps(progressive_instances, option=orjson.OPT_INDENT_2).decode("utf-8"))

        # extend the list of instances
        instances.extend(progressive_instances)
        createPlot(instances=progressive_instances, output=output, type=type, filename=file, mode="progressive", x="iteration", y="cosine similarity", extension=extension)
    return instances

#### BETWEEN FINALS VARIETY ####
def betweenFinalsVariety(input, type, extension, output, savejson=False):
    '''this function computes the similarity between each pair of final prompts, along with the similarity between the respective initial prompts
    path: the path containing the json files
    return: a list of dictionaries, each dictionary contains the final prompt 1, the final prompt 2, the cosine similarity between the two prompts, the initial prompt 1, the initial prompt 2, the initial cosine similarity between the two prompts, the configuration of the archive, the file name
    it also saves the result in a file betweenFinalsVarietySimilarity.json inside the folder betweenFinalsVariety in the input folder'''
    # select json files in the path
    files = glob.glob(input+"/*.json")
    if not os.path.exists(output):
        os.makedirs(output)

    instances = []
    for file in tqdm(files):
        print(f"analysing {file}")
        with open(file) as f:
            archive = Archive.from_dict(orjson.loads(f.read()))
            filename = os.path.basename(file)
            # remove the extension
            filename = os.path.splitext(filename)[0]
            for iteration in tqdm(range(archive.config.iterations)):
                for i in range(0, len(archive.runs)-1):
                    for j in range(i+1, len(archive.runs)):
                        instances.append({  "final prompt 1": archive.runs[i].taken[iteration].generated_prompt_for_sut,
                                            "final prompt 2": archive.runs[j].taken[iteration].generated_prompt_for_sut,
                                            "final cosine similarity": similarityBetween(model, 
                                                                                archive.runs[i].taken[iteration].generated_prompt_for_sut.lower().strip(), 
                                                                                archive.runs[j].taken[iteration].generated_prompt_for_sut.lower().strip()),
                                            "prompt from dataset 1": archive.runs[i].initial.prompt_from_dataset,
                                            "prompt from dataset 2": archive.runs[j].initial.prompt_from_dataset,
                                            "initial cosine similarity": similarityBetween(model, 
                                                                                archive.runs[i].initial.prompt_from_dataset.lower().strip(), 
                                                                                archive.runs[j].initial.prompt_from_dataset.lower().strip()),
                                            "config": archive.config, 
                                            "file name": filename, 
                                            "iteration": iteration})
            if savejson:
                with open(f"{output}/{filename}_betweenFinalsVariety.json", "w") as f:
                    f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))
            createPlot(instances=instances, output=output, type=type, filename=file, mode="betweenFinalsVariety", x="iteration", y="final cosine similarity", extension=extension)
    return instances

### SCORE CORRELATION ###

def scoreCorrelation(input, output, extension="png", savejson=False):
    '''this function computes the correlation between the delta cosine similarity and the score
    input: the path to the folder containing the json files
    output: the path to save the results
    extension: the extension of the output file
    return: a list of dictionaries, each dictionary contains the initial prompt, the final prompt, the delta cosine similarity, the initial score, the final score, the delta score, the configuration of the archive, the file name
    it also saves the result in a file scoreCorrelation.json inside the folder scoreCorrelation in the output folder and plots the correlation between the delta cosine similarity and the delta score, the delta cosine similarity and the final score, the initial score and the final score
    '''
    # select json files in the path
    files = glob.glob(f"{input}/*.json")
    instances = []

    # create the folder scoreCorrelation if it does not exist
    if not os.path.exists(output):
        os.makedirs(output)

    for file in tqdm(files):
        singleInstance = []
        print(f"analysing {file}")
        with open(file) as f:
            archive = Archive.from_dict(orjson.loads(f.read()))
            filename = os.path.basename(file)
            # remove the extension
            filename = os.path.splitext(filename)[0]
            for run in tqdm(archive.runs):
                singleInstance.append({  "initial prompt": run.initial.prompt_from_dataset,
                                        "final prompt": run.taken[-1].generated_prompt_for_sut,
                                        "delta cosine similarity": similarityBetween(model, 
                                                                                run.initial.prompt_from_dataset.lower().strip(), 
                                                                                run.taken[-1].generated_prompt_for_sut.lower().strip()),
                                        "initial score": max(run.initial.criterion.values()),
                                        "final score": max(run.taken[-1].criterion.values()),
                                        "delta score": max(run.taken[-1].criterion.values()) - max(run.initial.criterion.values()),
                                        "config": archive.config, 
                                        "fileName": filename})
            plotCorrelation(instances=singleInstance, title="Correlation between delta cosine similarity and delta score", x="delta cosine similarity", y="delta score", name=f"{filename}_correlation_deltaSim_deltaScore", output=output, extension=extension)
            plotCorrelation(instances=singleInstance, title="Correlation between delta cosine similarity and final score", x="delta cosine similarity", y="final score", name=f"{filename}_correlation_deltaSim_finalScore", output=output, extension=extension)
            plotCorrelation(instances=singleInstance, title="Correlation between initial and final score",                 x="initial score",           y="final score", name=f"{filename}_correlation_Score", output=output, extension=extension)
            if savejson:
                with open(f"{output}/{filename}_scoreCorrelation.json", "w") as f:
                    f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))
            instances.extend(singleInstance)
    
    return instances
    
def plotCorrelation(instances, output, title, x, y, name="correlation", extension="png"):
    # correlation between initial and cosine similarity, group by file name
    df = pd.DataFrame(instances)
    plt.figure(figsize=(20, 10))
    ax = sns.scatterplot(x=x, y=y, data=df, hue="fileName")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # save figure in file folder, with the type of plot, if it does not exist create it
    plt.savefig(f"{output}/{name}.{extension}", dpi=300, format=extension)
    plt.close()

### DISTANCE FROM INITIAL PROMPT ###
def distanceFromInitialPrompt(input, type, extension, output, savejson=False):
    files = glob.glob(input + "/*.json")
    print(files)
    data = []
    for file in tqdm(files):
        # filename without path and extension
        filename = os.path.splitext(os.path.basename(file))[0]
        with open(file, "r") as f:
            archive = Archive.from_dict(orjson.loads(f.read()))
            for run in tqdm(archive.runs):
                for iteration in range(0, len(run.taken)):
                    data.append({
                        "prompt_from_dataset": run.initial.prompt_from_dataset,
                        "generated_prompt": run.taken[iteration].generated_prompt_for_sut,
                        "cosine similarity": similarityBetween(model, run.initial.prompt_from_dataset, run.taken[iteration].generated_prompt_for_sut),
                        "config": archive.config,
                        "file name": filename,
                        "iteration": iteration
                    })
        path=os.path.dirname(input)
        # create file for similarity, inside the folder plots, in the input folder, if it does not exist create it
        if not os.path.exists(output):
            os.makedirs(output)
        if savejson:
            # save data to json
            with open(f"{output}/{filename}_sentenceDistance.json", "w") as f: 
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"))
        createPlot(instances=data, output=output, type=type, filename=filename, mode="distanceFromInitialPrompt", x="iteration", y="cosine similarity", extension=extension)
    return data

# global variable for the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@click.command()
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=False),
    help="Path to source file, the folder containing the output of evolutionary.py",
    required=True,
)
@click_option(
    "-dnc",
    "--donotcompute",
    is_flag=True,
    help="Don't compute similarity between sentences, just plot it",
    default=False,
)
@click_option(
    "-t",
    "--type",
    type=click.Choice(["violin", "boxplot"]),
    help="Type of plot to create: violin or boxplot. Default is boxplot",
    default="violin",
)
@click_option(
    "-m",
    "--mode",
    type=click.Choice(["progressive", "betweenFinalsVariety", "scoreCorrelation", "distanceFromInitialPrompt"]),
    help="""Mode of analysis: 'progressive': similarity between consecutive prompts, starting from the prompt from the dataset (saved in the folder progressive),  
            betweenFinalsVariety: similarity between final prompts, organised by iteration (saved in the folder betweenFinalsVariety),
            scoreCorrelation: correlation between cosine similarity and score
            distanceFromInitialPrompt: distance between the initial prompt and the generated prompt, organised by iteration (saved in the folder distanceFromInitialPrompt)
            """,
    default="progressive",
)
@click_option(
    "-e",
    "--extension",
    default="png",
    type=click.Choice(["png", "pdf", "svg"]),
    help="The extension of the output file",
)
@click_option(
    "-o",
    "--output",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
    help="Path to the output folder, where the plots will be saved",
    required=True,
)
@click_option(
    "-sj",
    "--savejson",
    is_flag=True,
    help="Save the results in a json file",
    default=False,
)
def main(input, donotcompute, type, mode, extension, output, savejson):
    '''this function is the main function of the script
    input: the path to the folder containing the files to analyse
    donotcompute: a boolean indicating if the analysis has already been done
    type: the type of plot to generate
    mode: the mode of the analysis
    '''
    if not os.path.exists(output):
        os.makedirs(output)

    outputFolder=output
    computedFiles = glob.glob(output+ "/*.json")
    outputFolder = output+"/"+mode+"/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if mode=="progressive":
        if donotcompute:
            instances=[]
            for computedFile in computedFiles :
                singleInstance = orjson.loads(open(computedFile).read())
                createPlot(instances=singleInstance, output=outputFolder, type=type, mode=mode, filename=computedFile, x="iteration", y="cosine similarity", extension=extension)
                instances.extend(singleInstance)
        else:
            instances = progressive(input, type, extension=extension, output=outputFolder, savejson=savejson)
        
        # create the general plot, showing the progressive cosine similarity between files, not by iteration
        createPlot(instances=instances,output=outputFolder, type=type,  mode=mode, filename="General Progressive Similarity", x="file name", y="cosine similarity", extension=extension)    
    elif mode=="betweenFinalsVariety":
        if donotcompute:
            instances=[]
            for computedFile in computedFiles :
                singleInstance = orjson.loads(open(computedFile).read())
                instances.extend(singleInstance)
        else:
            instances = betweenFinalsVariety(input, type, extension=extension, output=outputFolder, savejson=savejson)
        
        createPlot(instances=instances,output=outputFolder, type=type,  mode=mode, filename="general Between Finals Variety", x="file name", y="final cosine similarity", extension=extension)
    elif mode=="scoreCorrelation":
        if donotcompute:
            instances=[]
            for computedFile in computedFiles :
                singleInstance = orjson.loads(open(computedFile).read())
                instances.extend(singleInstance)
                filename = os.path.basename(computedFile)
                filename = filename.replace("_scoreCorrelation.json", "")
                plotCorrelation(instances=singleInstance, title="Correlation between delta cosine similarity and delta score", x="delta cosine similarity", y="delta score", name=f"{filename}_correlation_deltaSim_deltaScore", output=outputFolder, extension=extension)
                plotCorrelation(instances=singleInstance, title="Correlation between delta cosine similarity and final score", x="delta cosine similarity", y="final score", name=f"{filename}_correlation_deltaSim_finalScore", output=outputFolder, extension=extension)
                plotCorrelation(instances=singleInstance, title="Correlation between initial and final score",                 x="initial score",           y="final score", name=f"{filename}_correlation_Score", output=outputFolder, extension=extension)
        else:
            instances = scoreCorrelation(input, output=outputFolder, savejson=savejson, extension=extension)
        
        plotCorrelation(instances=instances, title="Correlation between delta cosine similarity and delta score", x="delta cosine similarity", y="delta score", name=f"General_correlation_deltaSim_deltaScore", output=outputFolder, extension=extension)
        plotCorrelation(instances=instances, title="Correlation between delta cosine similarity and final score", x="delta cosine similarity", y="final score", name=f"General_correlation_deltaSim_finalScore", output=outputFolder, extension=extension)
        plotCorrelation(instances=instances, title="Correlation between initial and final score",                 x="initial score",           y="final score", name=f"General_correlation_Score", output=outputFolder, extension=extension)
    elif mode=="distanceFromInitialPrompt":
        if donotcompute:
            instances=[]
            for computedFile in computedFiles :
                singleInstance = orjson.loads(open(computedFile).read())
                instances.extend(singleInstance)
                filename = os.path.basename(computedFile).replace("_sentenceDistance.json", "")
                createPlot(instances=singleInstance, output=outputFolder, type=type, mode=mode, filename=filename, x="iteration", y="cosine similarity", extension=extension)
        else:
            instances = distanceFromInitialPrompt(input, type, extension=extension, output=outputFolder, savejson=savejson)
        createPlot(instances=instances,output=outputFolder, type=type,  mode=mode, filename="general", x="file name", y="cosine similarity", extension=extension)


if __name__ == "__main__":
    main()