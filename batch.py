import os
import glob
import sys

import click
from tqdm import tqdm
from comparison import get_files


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


def run_command(command):
    print(f"Running command: {command}")
    os.system(command)


@click.command()
@click_option(
    "--all",
    is_flag=True,
    help="Run all the analysis",
    default=False,
)
@click_option(
    "--chat-analysis",
    is_flag=True,
    help="Run the chat analysis",
    default=False,
)
@click_option(
    "--comparison",
    is_flag=True,
    help="Run the comparison",
    default=False,
)
@click_option(
    "--similarity-aggregate",
    is_flag=True,
    help="Run the sentenceAnalyser for the folder",
    default=False,
)
@click_option(
    "--similarity-single",
    is_flag=True,
    help="Run the sentenceAnalyser for the folder in a single list",
    default=False,
)
@click_option(
    "--distance",
    is_flag=True,
    help="Run the distanceSentenceAnalyser for all the files in the folder",
    default=False,
)
@click_option(
    "--general-sentence-analysis",
    is_flag=True,
    help="Run the all sentence analysis from sentenceAnalyser",
    default=False,
)
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=False),
    required=True,
    help="The path to the results folder",
)
@click_option(
    "-o",
    "--output",
    type=click.Path(exists=False, resolve_path=True, dir_okay=True, file_okay=False),
    required=False,
    help="The path to the results folder",
)
@click_option(
    "--donotcompute",
    "-dnc",
    is_flag=True,
    help="Do not compute the results, just plot them",
    default=False,
)
@click_option(
    "-e",
    "--extension",
    default="svg",
    type=click.Choice(["png", "pdf", "svg"]),
    help="The extension of the output file",
)
@click_option(
    "-gc",
    "--generalcomparison",
    is_flag=True,
    help="Run the generalComparison script",
    default=False,
)
@click_option(
    "-wc",
    "--wordcloud",
    is_flag=True,
    help="Run the wordcloud script",
    default=False,
)
@click_option(
    "-sj",
    "--savejson",
    is_flag=True,
    help="Save the json files",
    default=False,
)
@click_option(
    "-cd",
    "--contextDivergence",
    is_flag=True,
    help="Run the context divergence script",
    default=False,
)
def main(
    all,
    chat_analysis: bool,
    comparison: bool,
    similarity_aggregate: bool,
    similarity_single: bool,
    distance: bool,
    general_sentence_analysis: bool,
    input: str,
    output: str,
    donotcompute: bool,
    extension: str,
    generalcomparison: bool,
    wordcloud: bool,
    savejson: bool,
    contextdivergence: bool,
):
    if all:
        chat_analysis = True
        comparison = True
        similarity_aggregate = True
        similarity_single = True
        distance = True
        general_sentence_analysis = True
        generalcomparison = True
        wordcloud = True
        contextdivergence = True

    criteria = ["max", "min", "avg", "median"]
    if output is None:
        output = input
    
    if not os.path.exists(output):
        os.makedirs(output)

    donotcompute = "-dnc" if donotcompute else ""
    savejson = "-sj" if savejson else ""

    # replace .par files with .json
    partialFiles = glob.glob(input + "/*.par")
    for file in partialFiles:
        os.rename(file, file.replace(".par", ""))


    if chat_analysis:
        print("Running chatAnalysis")
        # for each file in the folder results, run the analysis plot with the file path as argument
        files = get_files(input, includeBaseline=False)
        if not os.path.exists(output + "/chatAnalysis"):
            os.makedirs(output + "/chatAnalysis")
        for file in tqdm(files, desc="Files", position=0, leave=True, file=sys.stdout):
            # if file is without an extension, or has .par, rename it to .json
            if file.find(".par") != -1:
                os.rename(file, file.replace(".par", ".json"))
                file = file.replace(".par", ".json")
            else:
                # if file has no extension, add .json
                if file.find(".") == -1:
                    os.rename(file, file + ".json")
                    file = file + ".json"

            for c in tqdm(
                criteria, desc="Criteria", position=0, leave=True, file=sys.stdout
            ):
                run_command(
                    f"python3 chatAnalysis.py --silent --input {file} --output {output}/chatAnalysis --criteria {c} --extension {extension}"
                )
    if comparison:
        print("Running comparison")

        if not os.path.exists(output + "/comparisons"):
            os.makedirs(output + "/comparisons")
        for c in tqdm(
            criteria, desc="Criteria", position=0, leave=True, file=sys.stdout
        ):
            run_command(
                f"python3 comparison.py --silent --input {input} --output {output}/comparisons --criteria {c} {donotcompute} --extension {extension}"
            )

    # if distance:
    #     print("Running distanceSentenceAnalyser")
    #     if not os.path.exists(output + "/distance"):
    #         os.makedirs(output + "/distance")
    #         run_command(
    #             f"python distanceSentenceAnalyser.py --input {input} --output {output}/distance --type violin"
    #         )

    if general_sentence_analysis:
        print("Running generalSentenceAnalysis")
        for m in tqdm(
            ["progressive", "betweenFinalsVariety", "scoreCorrelation", "distanceFromInitialPrompt"],
            desc="Modes",
            position=0,
            leave=True,
            file=sys.stdout,
        ):
            run_command(
                f"python3 sentenceAnalyser.py --input {input} --mode {m} --type violin {donotcompute} -e {extension} -o {output}/sentenceAnalysis {savejson}"
            )
        
    if generalcomparison:
        print("Running generalComparison")
        # input folder for the general comparison is the input path without the last folder
        inputGC = os.path.dirname(input)

        #output folder is the same as the input folder
        outputGC = inputGC

        for type in ["boxplot", "violin", "line"]:
            run_command(
                f"python3 generalComparison.py --input {inputGC} --output {outputGC} --type {type} -e {extension}"
            )
            if type != "line":
                run_command(
                    f"python3 generalComparison.py --input {inputGC} --output {outputGC} --type {type} -e {extension} -m"
                )
        for analysis in ["time"]:
            for groupby in ["sut", "sg"]:
                run_command(
                    f"python3 generalComparison.py --input {inputGC} --output {outputGC} --type boxplot -e {extension} --analysis {analysis} --groupby {groupby} -m"
                )
            

    
    if wordcloud:
        print("Running wordcloud")
        run_command(
            f"python3 clouds.py --input {input} --output {output} --extension {extension}"
        )
    
    if contextdivergence:
        print("Running context divergence")
        # input folder for the general comparison is the input path without the last folder
        inputGC = os.path.dirname(input)

        #output folder is the same as the input folder
        outputGC = inputGC
        run_command(
            f"python3 tools/contextDivergence.py --input {inputGC} --output {outputGC} --extension {extension}"
        )

    
        
    



if __name__ == "__main__":
    main()
