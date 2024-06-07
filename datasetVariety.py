
import orjson
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import click
from sentenceAnalyser import similarityBetween, model

def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

### DATASET VARIETY ###
def datasetVariety(dataset):
    '''this function computes the similarity between each pair of prompts in the dataset
    dataset: the dataset containing the prompts
    return: a list of dictionaries, each dictionary contains the prompt 1, the prompt 2, the cosine similarity between the two prompts
    it also saves the result in a file datasetVarietySimilarity.json inside the folder datasetVariety in the input folder'''
    instances = []

    file = open(dataset)
    questions = []
    for line in file.readlines():
        questions.append(line)
    # make it multi threaded to speed up the process
    from concurrent.futures import ThreadPoolExecutor, as_completed
    LENGTH = int((len(questions))*(len(questions)-1)/2) # Number of iterations required to fill pbar
    pbar = tqdm(total=LENGTH, desc='remaining')  # Init pbar
    instances = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(questions)-1):
            instances.extend(executor.submit(instanceSimilarity, model, questions[i], questions[j]) for j in range(i+1, len(questions)))
        for _ in as_completed(instances):
            pbar.update(n=1)  # Increments counter
    instances = [i.result() for i in instances]
    
    path=os.path.dirname(dataset)
    dataset=os.path.basename(dataset)
    # save the similarity in the file similarity.json
    with open(f"{path}/datasetVariety/{dataset}_datasetVarietySimilarity.json", "w") as f:
        f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))

    return instances

def instanceSimilarity(model, prompt1, prompt2):
    '''this function computes the cosine similarity between two prompts
    model: the model used to compute the embeddings
    prompt1: the first prompt
    prompt2: the second prompt
    return: the cosine similarity between the two prompts'''
    instance={
        "prompt 1": prompt1,
        "prompt 2": prompt2,
        "cosine similarity": similarityBetween(model, prompt1.lower().strip(), prompt2.lower().strip())
    }
    return instance

def plotDatasetVariety(instances, datasetName, output, normalise=False, instances2=None, datasetName2=None, extension="png"):
    '''this function plots the distribution of the similarity between the prompts in the dataset
    instances: the list of dictionaries containing the prompt 1, the prompt 2, the cosine similarity between the two prompts
    datasetName: the name of the dataset
    output: the path to save the plot
    normalise: if True, normalise the count of values between 0 and 1
    instances2: the list of dictionaries containing the prompt 1, the prompt 2, the cosine similarity between the two prompts for the second dataset(if provided)
    datasetName2: the name of the second dataset(if provided)'''
    df=pd.DataFrame(instances)
    df["dataset"]=datasetName
    plt.figure(figsize=(15, 6))

    if instances2:
        df2=pd.DataFrame(instances2)
        df2["dataset"]=datasetName2
        df=pd.concat([df, df2])
    if normalise:
        sns.displot(data=df, x="cosine similarity", hue="dataset", stat="probability", common_norm=False, multiple="dodge")
        title=f"Normalised similarity between {datasetName} and {datasetName2}"
    else:
        sns.displot(data=df, x="cosine similarity", hue="dataset", multiple="dodge")
        title=f"Similarity between {datasetName} and {datasetName2}"
    
    plt.title(title)
    plt.savefig(output+"."+extension, bbox_inches='tight', dpi=300, format=extension)





@click.command()

@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=True),
    help="Path to source file, the dataset containing the prompts, in a text file one prompt per line, or if do not compute is True, the file containing the similarity between the prompts",
    required=True,
)

@click_option(
    "-i2",
    "--input2",
    type=click.Path(exists=False, resolve_path=False, dir_okay=True, file_okay=True),
    help="Path to another source file, the dataset containing the prompts in a text file, one prompt per line, or if do not compute is True, the file containing the similarity between the prompts",
    required=False,
)


@click_option(
    "-dnc",
    "--donotcompute",
    is_flag=True,
    help="Don't compute similarity between sentences, just plot it, the input file should contain the similarity between the prompts (for both input and input2 if provided)",
    default=False,
)
@click_option(
    "-n",
    "--normalise",
    is_flag=True,
    help="Normalise the count of values between 0 and 1",
    default=False,
)
@click_option(
    "-e",
    "--extension",
    default="png",
    type=click.Choice(["png", "pdf", "svg"]),
    help="The extension of the output file",
)
def main(input, input2, donotcompute,normalise, extension):
    '''this function computes the similarity between each pair of prompts in the dataset
    input: the dataset containing the prompts
    donotcompute: if True, it does not compute the similarity between the prompts, just plot it
    '''
    path=os.path.dirname(input)
    datasetName=os.path.basename(input).replace(".json", "").replace("_datasetVarietySimilarity", "")

    if input2:
        path2=os.path.dirname(input2)
        datasetName2=os.path.basename(input2).replace(".json", "").replace("_datasetVarietySimilarity", "")

    if not os.path.exists(path+"/datasetVariety"):
        os.makedirs(path+"/datasetVariety")
    
    if not donotcompute:
        instances=datasetVariety(input)
        if input2:
            instances2=datasetVariety(input2)
    else:
        with open(input) as f:
            instances=orjson.loads(f.read())
        if input2:
            with open(input2) as f:
                instances2=orjson.loads(f.read())
    
    if input2:
        plotDatasetVariety(instances=instances, output=f"{path}/datasetVariety/compared_{datasetName}_{datasetName2}_datasetVarietySimilarity", normalise=normalise, instances2=instances2, datasetName=datasetName, datasetName2=datasetName2, extension=extension)
    else:
        plotDatasetVariety(instances=instances, output=f"{path}/datasetVariety/{datasetName}_datasetVarietySimilarity", normalise=normalise, datasetName=datasetName, extension=extension)
    
if __name__ == "__main__":
    main()