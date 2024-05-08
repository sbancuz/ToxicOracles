from sentence_transformers import SentenceTransformer


from evolutionary import Archive


import orjson
import click
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sentenceAnalyser import distancebetween


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


@click.command()
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=False),
    help="Path to source file, the output of evolutionary.py (output.json)",
)
@click_option(
    "-o",
    "--output",
    type=click.Path(exists=False, resolve_path=True, dir_okay=True, file_okay=False),
    help="Path to output folder",
)
@click_option(
    "-t",
    "--type",
    type=click.Choice(["violin", "boxplot"]),
    help="Type of plot to create: violin or boxplot. Default is boxplot",
    default="boxplot",
)
def main(input, output, type):
    files = glob.glob(input + "/*.json")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    data = []
    for file in tqdm(files):
        # filename without path and extension
        filename = os.path.splitext(os.path.basename(file))[0]
        with open(file, "r") as f:
            archive = Archive.from_dict(orjson.loads(f.read()))
            for run in tqdm(archive.runs):
                for iteration in tqdm(range(0, len(run.taken))):
                    data.append(
                        {
                            "prompt_from_dataset": run.initial.prompt_from_dataset,
                            "generated_prompt": run.taken[
                                iteration
                            ].generated_prompt_for_sut,
                            "distance": distancebetween(
                                model,
                                run.initial.prompt_from_dataset,
                                run.taken[iteration].generated_prompt_for_sut,
                            ),
                            "config": archive.config,
                            "filename": filename,
                            "iteration": iteration,
                        }
                    )

        # save data to json
        with open(
            output + "/" + filename + "sentenceDistance.json",
            "w",
        ) as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"))
        plot_data = pd.DataFrame(data)

        if type == "violin":
            sns.violinplot(x="iteration", y="distance", data=plot_data)
        else:
            sns.boxplot(x="iteration", y="distance", data=plot_data)

        plt.savefig(
            output + "/" + filename + "sentenceDistance.png",
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    main()
