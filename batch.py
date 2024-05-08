import os
import glob
import sys

import click
from tqdm import tqdm


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
    required=True,
    help="The path to the results folder",
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
):
    if all:
        chat_analysis = True
        comparison = True
        similarity_aggregate = True
        similarity_single = True
        distance = True
        general_sentence_analysis = True

    criteria = ["max", "min", "avg", "median"]
    if not os.path.exists(output):
        os.makedirs(output)

    if chat_analysis:
        print("Running chatAnalysis")
        # for each file in the folder results, run the analysis plot with the file path as argument
        files = glob.glob(input + "/*")
        if not os.path.exists(output + "/chatAnalysis"):
            os.makedirs(output + "/chatAnalysis")
        for file in tqdm(files, desc="Files", position=0, leave=True, file=sys.stdout):
            # exlude file if has baseline in the name or it is a directory
            if "baseline" not in file and not os.path.isdir(file):
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
                        f"python chatAnalysis.py --silent --input {file} --output {output}/chatAnalysis --criteria {c}"
                    )
    if comparison:
        print("Running comparison")

        if not os.path.exists(output + "/comparisons"):
            os.makedirs(output + "/comparisons")
        for c in tqdm(
            criteria, desc="Criteria", position=0, leave=True, file=sys.stdout
        ):
            run_command(
                f"python comparison.py --silent --input {input} --output {output}/comparisons --criteria {c}"
            )

    if distance:
        print("Running distanceSentenceAnalyser")
        if not os.path.exists(output + "/distance"):
            os.makedirs(output + "/distance")
            run_command(
                f"python distanceSentenceAnalyser.py --input {input} --output {output}/distance --type violin"
            )

    ## NON FUNZIONANO
    # if general_sentence_analysis:
    #     print("Running generalSentenceAnalysis")
    #     # ["progressive", "finalVariety", "scoreCorrelation", "datasetVariety"],
    #     for m in tqdm(
    #         ["progressive"],
    #         desc="Modes",
    #         position=0,
    #         leave=True,
    #         file=sys.stdout,
    #     ):
    #         # run_command(
    #         #     f"python sentenceAnalyser.py --input {input} --mode {m} --type violin"
    #         # )
    #
    #         run_command(
    #             f"python sentenceAnalyser.py --donotcompute --input {input} --mode {m} --type boxplot"
    #         )

    # if similarity_aggregate:


if __name__ == "__main__":
    main()
