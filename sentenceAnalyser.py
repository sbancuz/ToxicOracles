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


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


def distancebetween(model, sentence1, sentence2):
    """this function computes the cosine similarity between two sentences
    model: the model used to compute the embeddings
    sentence1: the first sentence
    sentence2: the second sentence
    return: the cosine similarity between the two sentences
    """

    # Compute embedding for both lists
    embedding_1 = model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)
    # Compute cosine-similarits
    cosine_scores = float((util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]))
    return cosine_scores


def analyse_similarity(file):
    """this function analyses the progressive similarity between prompts in the file, starting from the prompt from the dataset
    file: the file containing the archive
    return: a list of dictionaries, each dictionary contains the starting prompt, the generated prompt, the final cosine similarity between the two prompts, the configuration of the archive, the file name and the iteration
    it also saves the result in a file filename+progressivesimilarity.json inside the folder progressiveSimilarity in the folder of the file
    """
    archive: Archive
    instances = []
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with open(file) as f:
        archive = Archive.from_dict(orjson.loads(f.read()))

    for run in tqdm(archive.runs):
        for i in tqdm(range(0, len(run.taken))):
            filename = os.path.basename(file)
            # remove the extension
            filename = os.path.splitext(filename)[0]
            if i == 0:
                if (
                    run.initial.prompt_from_dataset.lower().strip()
                    != run.taken[i].generated_prompt_for_sut.lower().strip()
                ):
                    instances.append(
                        {
                            "starting": run.initial.prompt_from_dataset,
                            "generated": run.taken[i].generated_prompt_for_sut,
                            "final cosine similarity": distancebetween(
                                model,
                                run.initial.prompt_from_dataset,
                                run.taken[i].generated_prompt_for_sut,
                            ),
                            "config": archive.config,
                            "fileName": filename,
                            "iteration": i,
                        }
                    )
            else:
                if (
                    run.taken[i - 1].generated_prompt_for_sut.lower().strip()
                    != run.taken[i].generated_prompt_for_sut.lower().strip()
                ):
                    # instances.append({"starting": run.taken[i - 1].generated_prompt_for_sut, "generated": run.taken[i].generated_prompt_for_sut, "cosine similarity": float(util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]), "config": archive.config, "fileName": filename})
                    instances.append(
                        {
                            "starting": run.taken[i - 1].generated_prompt_for_sut,
                            "generated": run.taken[i].generated_prompt_for_sut,
                            "final cosine similarity": distancebetween(
                                model,
                                run.taken[i - 1].generated_prompt_for_sut,
                                run.taken[i].generated_prompt_for_sut,
                            ),
                            "config": archive.config,
                            "fileName": filename,
                            "iteration": i,
                        }
                    )
    # save the similarity in the file filename+progressivesimilarity.json
    # filename without path and extension
    filename = os.path.splitext(os.path.basename(file))[0]
    # save in the path of the file + progressive similarity in file filename+progressivesimilarity.json
    if not os.path.exists(f"{os.path.dirname(file)}/progressiveSimilarity"):
        os.makedirs(f"{os.path.dirname(file)}/progressiveSimilarity")
    with open(
        f"{os.path.dirname(file)}/progressiveSimilarity/{filename}progressiveSimilarity.json",
        "w",
    ) as f:
        f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))

    return instances


def progressive(files, input):
    """this function computes the progressive similarity between the prompts inside a list of files
    files: the list of files containing the archives
    input: the input folder
    return: a list of dictionaries, each dictionary contains the starting prompt, the generated prompt, the final cosine similarity between the two prompts, the configuration of the archive, the file name and the iteration of all the files
    it also saves the result in a file progressiveSimilarity.json inside the folder progressiveSimilarity in the input folder"""
    instances = []
    # iterate over files
    for file in files:
        if os.path.isdir(file):
            continue
        print(f"analysing {file}")
        instances.extend(analyse_similarity(file))

    # sort instances by similarity
    # instances.sort(key=lambda x: x["final cosine similarity"], reverse=True)
    # create file for similarity, inside the folder plots, in the input folder, if it does not exist create it
    if not os.path.exists(f"{input}/progressiveSimilarity"):
        os.makedirs(f"{input}/progressiveSimilarity")
    # save the similarity in the file similarity.json
    with open(f"{input}/progressiveSimilarity/progressiveSimilarity.json", "w") as f:
        f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))
    createBP(
        instances,
        input + "/" + "progressiveSimilarity",
        "violin",
        "progressive",
        "Distance between consecutive generated prompts",
    )

    return instances


def betweenFinalsVariety(path):
    """this function computes the similarity between each pair of final prompts, along with the similarity between the respective initial prompts
    path: the path containing the json files
    return: a list of dictionaries, each dictionary contains the final prompt 1, the final prompt 2, the final cosine similarity between the two prompts, the initial prompt 1, the initial prompt 2, the initial cosine similarity between the two prompts, the configuration of the archive, the file name
    it also saves the result in a file finalVarietySimilarity.json inside the folder betweenFinalsVariety in the input folder"""
    # select json files in the path
    files = glob.glob(f"{path}/*.json")
    instances = []
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for file in tqdm(files):
        print(f"analysing {file}")
        with open(file) as f:
            archive = Archive.from_dict(orjson.loads(f.read()))
            filename = os.path.basename(file)
            # remove the extension
            filename = os.path.splitext(filename)[0]
            for i in tqdm(range(0, len(archive.runs) - 1)):
                for j in tqdm(range(i + 1, len(archive.runs))):
                    instances.append(
                        {
                            "final prompt 1": archive.runs[i]
                            .taken[-1]
                            .generated_prompt_for_sut,
                            "final prompt 2": archive.runs[j]
                            .taken[-1]
                            .generated_prompt_for_sut,
                            "final cosine similarity": distancebetween(
                                model,
                                archive.runs[i]
                                .taken[-1]
                                .generated_prompt_for_sut.lower()
                                .strip(),
                                archive.runs[j]
                                .taken[-1]
                                .generated_prompt_for_sut.lower()
                                .strip(),
                            ),
                            "prompt from dataset 1": archive.runs[
                                i
                            ].initial.prompt_from_dataset,
                            "prompt from dataset 2": archive.runs[
                                j
                            ].initial.prompt_from_dataset,
                            "initial cosing distance": distancebetween(
                                model,
                                archive.runs[i]
                                .initial.prompt_from_dataset.lower()
                                .strip(),
                                archive.runs[j]
                                .initial.prompt_from_dataset.lower()
                                .strip(),
                            ),
                            "config": archive.config,
                            "fileName": filename,
                        }
                    )
    # sort instances by similarity
    instances.sort(key=lambda x: x["final cosine similarity"], reverse=True)

    # create file for similarity, inside the folder plots, in the input folder, if it does not exist create it
    if not os.path.exists(f"{path}/betweenFinalsVariety"):
        os.makedirs(f"{path}/betweenFinalsVariety")
    # save the similarity in the file similarity.json
    with open(f"{path}/betweenFinalsVariety/finalVarietySimilarity.json", "w") as f:
        f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))
    createBP(
        instances,
        path + "/" + "betweenFinalsVariety",
        "violin",
        "finalVariety",
        "Distance between final generated prompts",
    )
    return instances


def scoreCorrelation(path):
    """this function computes the correlation between the delta cosine similarity and the score
    path: the path containing the json files
    return: a list of dictionaries, each dictionary contains the initial cosine similarity, the final cosine similarity, the delta cosine similarity, the score, the configuration of the archive, the file name
    it also saves the result in a file scoreCorrelation.json inside the folder scoreCorrelation in the input folder"""
    # select json files in the path
    files = glob.glob(f"{path}/*.json")
    instances = []
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for file in tqdm(files):
        print(f"analysing {file}")
        with open(file) as f:
            archive = Archive.from_dict(orjson.loads(f.read()))
            filename = os.path.basename(file)
            # remove the extension
            filename = os.path.splitext(filename)[0]
            for run in tqdm(archive.runs):
                instances.append(
                    {
                        "initial prompt": run.initial.prompt_from_dataset,
                        "final prompt": run.taken[-1].generated_prompt_for_sut,
                        "delta cosine similarity": distancebetween(
                            model,
                            run.initial.prompt_from_dataset.lower().strip(),
                            run.taken[-1].generated_prompt_for_sut.lower().strip(),
                        ),
                        "initial score": max(run.initial.criterion.values()),
                        "final score": max(run.taken[-1].criterion.values()),
                        "delta score": max(run.taken[-1].criterion.values())
                        - max(run.initial.criterion.values()),
                        "config": archive.config,
                        "fileName": filename,
                    }
                )
    # sort instances by similarity
    instances.sort(key=lambda x: x["delta cosine similarity"], reverse=True)

    # create file for similarity, inside the folder plots, in the input folder, if it does not exist create it
    if not os.path.exists(f"{path}/scoreCorrelation"):
        os.makedirs(f"{path}/scoreCorrelation")
    # save the similarity in the file similarity.json
    with open(f"{path}/scoreCorrelation/scoreCorrelation.json", "w") as f:
        f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))
    plotCorrelation(
        instances,
        path,
        "Correlation between delta cosine similarity and delta score",
        "delta cosine similarity",
        "delta score",
        "scoreCorrelation",
        "correlation_deltaSim_deltaScore.png",
    )
    plotCorrelation(
        instances,
        path,
        "Correlation between delta cosine similarity and final score",
        "delta cosine similarity",
        "final score",
        "scoreCorrelation",
        "correlation_deltaSim_finalScore.png",
    )
    plotCorrelation(
        instances,
        path,
        "Correlation between initial and final score",
        "initial score",
        "final score",
        "scoreCorrelation",
        "correlation_Score.png",
    )


def instanceSimilarity(model, prompt1, prompt2):
    """this function computes the cosine similarity between two prompts
    model: the model used to compute the embeddings
    prompt1: the first prompt
    prompt2: the second prompt
    return: the cosine similarity between the two prompts"""
    instance = {
        "prompt 1": prompt1,
        "prompt 2": prompt2,
        "cosine similarity": distancebetween(
            model, prompt1.lower().strip(), prompt2.lower().strip()
        ),
    }
    return instance


def datasetVariety(dataset):
    """this function computes the similarity between each pair of prompts in the dataset
    dataset: the dataset containing the prompts
    return: a list of dictionaries, each dictionary contains the prompt 1, the prompt 2, the cosine similarity between the two prompts
    it also saves the result in a file datasetVarietySimilarity.json inside the folder datasetVariety in the input folder"""
    instances = []
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    file = open(dataset)
    questions = []
    for line in file.readlines():
        questions.append(line)

    # make it multi threaded to speed up the process
    from concurrent.futures import ThreadPoolExecutor, as_completed

    LENGTH = int(
        (len(questions)) * (len(questions) - 1) / 2
    )  # Number of iterations required to fill pbar
    pbar = tqdm(total=LENGTH, desc="remaining")  # Init pbar
    instances = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(questions) - 1):
            instances.extend(
                executor.submit(instanceSimilarity, model, questions[i], questions[j])
                for j in range(i + 1, len(questions))
            )
        for _ in as_completed(instances):
            pbar.update(n=1)  # Increments counter
    instances = [i.result() for i in instances]

    """'
    for i in tqdm(range(0, len(questions)-1)):
        for j in tqdm(range(i+1, len(questions))):
            instances.append({  "prompt 1": questions[i],
                                "prompt 2": questions[j],
                                "cosine similarity": distancebetween(model, 
                                                                     questions[i].lower().strip(), 
                                                                     questions[j].lower().strip())})
    #sort instances by similarity 
    instances.sort(key=lambda x: x["cosine similarity"], reverse=True)
    """

    path = os.path.dirname(dataset)
    dataset = os.path.basename(dataset)
    # create file for similarity, inside the folder plots, in the input folder, if it does not exist create it
    if not os.path.exists(f"{path}/datasetVariety"):
        os.makedirs(f"{path}/datasetVariety")
    # save the similarity in the file similarity.json
    with open(f"{path}/datasetVariety/datasetVarietySimilarity.json", "w") as f:
        f.write(orjson.dumps(instances, option=orjson.OPT_INDENT_2).decode("utf-8"))
        # create an histogram for the similarity
        df = pd.DataFrame(instances)
        plt.figure(figsize=(20, 10))
        ax = sns.histplot(df["cosine similarity"])
        ax.set_title("Cosine similarity between prompts in the dataset " + dataset)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Frequency")
        plt.savefig(
            f"{path}/datasetVariety/{dataset}_datasetVarietySimilarity.png", dpi=300
        )

    return instances


@click.command()
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=True),
    help="Path to source file, the output of evolutionary.py (output.json)",
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
    default="boxplot",
)
@click_option(
    "-m",
    "--mode",
    type=click.Choice(
        ["progressive", "finalVariety", "scoreCorrelation", "datasetVariety"]
    ),
    help="Mode of analysis: progressive or finalVariety. Default is progressive",
    default="progressive",
)
def main(input, donotcompute, type, mode):
    files = glob.glob(input + "/*.json")
    print(files)
    if donotcompute:
        # read file
        if mode == "progressive":
            # plot each file
            for file in files:
                with open(file) as f:
                    instances = orjson.loads(f.read())
                    createSingleBP(instances, input, type, mode, file)
            with open(f"{input}/progressiveSimilarity/progressiveSimilarity.json") as f:
                instances = orjson.loads(f.read())
        elif mode == "finalVariety":
            with open(f"{input}/betweenFinalsVariety/finalVarietySimilarity.json") as f:
                instances = orjson.loads(f.read())
        elif mode == "scoreCorrelation":
            with open(f"{input}/scoreCorrelation/scoreCorrelation.json") as f:
                instances = orjson.loads(f.read())
            plotCorrelation(
                instances,
                input,
                "Correlation between delta cosine similarity and delta score",
                "delta cosine similarity",
                "delta score",
                "scoreCorrelation",
                "correlation_deltaSim_deltaScore.png",
            )
            plotCorrelation(
                instances,
                input,
                "Correlation between delta cosine similarity and final score",
                "delta cosine similarity",
                "final score",
                "scoreCorrelation",
                "correlation_deltaSim_finalScore.png",
            )
            plotCorrelation(
                instances,
                input,
                "Correlation between initial and final score",
                "initial score",
                "final score",
                "scoreCorrelation",
                "correlation_Score.png",
            )
        elif mode == "datasetVariety":
            with open(f"{input}/datasetVariety/datasetVarietySimilarity.json") as f:
                instances = orjson.loads(f.read())

    else:
        if mode == "progressive":
            instances = progressive(files, input)
            createBP(
                instances,
                input,
                type,
                mode,
                "Distance between consecutive generated prompts",
            )
        elif mode == "finalVariety":
            instances = betweenFinalsVariety(input)
            createBP(
                instances, input, type, mode, "Distance between final generated prompts"
            )
        elif mode == "scoreCorrelation":
            instances = scoreCorrelation(input)
        elif mode == "datasetVariety":
            instances = datasetVariety(input)
    if mode == "finalVariety":
        plotCorrelation(
            instances,
            input,
            "Correlation between initial and final cosine similarity",
            "initial cosine similarity",
            "final cosine similarity",
            mode,
        )


def plotCorrelation(instances, input, title, x, y, mode, name="correlation.png"):
    # correlation between initial and final cosine similarity, group by file name
    df = pd.DataFrame(instances)
    plt.figure(figsize=(20, 10))
    ax = sns.scatterplot(x=x, y=y, data=df, hue="fileName")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.savefig(f"{input}/{mode}/{name}", dpi=300)
    plt.show()


def createBP(instances, input, type, mode, title):
    df = pd.DataFrame(instances)
    # violin plot, divided by file name
    plt.figure(figsize=(20, 10))
    if type == "boxplot":
        ax = sns.boxplot(x="fileName", y="final cosine similarity", data=df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(title)
        ax.set_xlabel("File name")
        ax.set_ylabel("Cosine similarity")
    else:
        ax = sns.violinplot(x="fileName", y="final cosine similarity", data=df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(title)
        ax.set_xlabel("File name")
        ax.set_ylabel("Cosine similarity")
    # save figure in file folder, with the type of plot
    plt.savefig(f"{input}/{mode}_similarity_{type}.png", dpi=300)
    plt.close()


def createSingleBP(instances, input, type, mode, file):
    # file name without path and extension and removing progressiveSimilarity from the name
    filename = os.path.basename(file)
    # remove the extension
    filename = os.path.splitext(filename)[0]
    # remove progressiveSimilarity from the name
    filename = filename.replace("progressiveSimilarity", "")

    df = pd.DataFrame(instances)
    # violin plot, divided by file name
    plt.figure(figsize=(20, 10))
    if type == "boxplot":
        ax = sns.boxplot(x="iteration", y="final cosine similarity", data=df)
        ax.set_title("Cosine similarity between sentences of " + filename)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine similarity")
    else:
        ax = sns.violinplot(x="iteration", y="final cosine similarity", data=df)
        ax.set_title("Cosine similarity between sentences of " + filename)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine similarity")
    # save figure in file folder, with the type of plot
    plt.savefig(f"{input}/{filename}_similarity_{type}.png", dpi=300)
    plt.close()


# global variable for the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()

