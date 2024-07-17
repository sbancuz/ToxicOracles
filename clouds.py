# create a wordcloud from the generated prompt for SUT
from evolutionary import Archive
import orjson
import os
import matplotlib.pyplot as plt
from typing import List
import click
from wordcloud import WordCloud
from comparison import parse_config

def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

@click.command()
@click_option(
    "--input",
    "-i",
    help="The path to the folder containing the results of the experiments",
    type=click.Path(exists=True),
    required=True,
    default="results/finalResults/vicunaUC_vicunaUC",
)
@click_option(
    "--output",
    "-o",
    help="The path to the output folder",
    type=click.Path(exists=True),
    required=True,
    default="results/finalResults/vicunaUC_vicunaUC/analysis",
)
@click_option(
    "--extension",
    "-e",
    help="The extension of the output file",
    default="svg",
    type=click.Choice(["png", "jpg", "svg", "pdf"]),
)
@click_option(
    "--verbose",
    "-v",
    help="Do show the plot",
    is_flag=True
)
def wordcloud(input, output, extension, verbose):
    '''
    This function is used to create a wordcloud from the generated prompt for SUT
    input: the path to the folder containing the results of the experiments
    output: the path to the output folder
    '''
    outputfolder = output + "/wordclouds/"
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    files = [os.path.join(input, f) for f in os.listdir(input) if f.endswith(".json") and f != "baseline.json"]
    for file in files:
        with open(file) as f:
            data = Archive.from_dict(orjson.loads(f.read()))
            prompt = ""
            promptCategory = {}
            for run in data.runs:
                for iteration in run.taken :
                        prompt += iteration.generated_prompt_for_sut + " "
                        if iteration.category not in promptCategory.keys():
                            promptCategory[iteration.category]=""
                        else:
                            promptCategory[iteration.category]+=iteration.generated_prompt_for_sut + " "
            wordcloud = WordCloud(max_font_size=200, max_words=200, background_color="white", width=1920, height=1080).generate(prompt)
            plt.figure(figsize=(20, 10))
            plt.title("Wordcloud of generated prompts for SUT: "+data.config.system_under_test+" by prompt generator "+data.config.prompt_generator+" ("+parse_config(data.config)+")", fontsize=20)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            filename = file.split("/")[-1].split(".")[0]
            plt.savefig(outputfolder + filename + "."+extension, dpi=300, format=extension)
            if verbose:
                plt.show()
            
            # compute the wordcloud for each element in promptCategory
            for element in promptCategory.keys():
                #print(promptCategory.keys())
                # create wordcloud
                prompts=promptCategory.get(element)
                if(len(prompts)>0):
                    wordcloud = WordCloud(max_font_size=200, max_words=200, background_color="white", width=1920, height=1080).generate(prompts)
                    plt.figure(figsize=(20, 10))
                    plt.title("Wordcloud of generated "+element+" prompts for S(L)UT: "+data.config.system_under_test+" by prompt generator "+data.config.prompt_generator+" ("+parse_config(data.config)+")", fontsize=20)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    filename = file.split("/")[-1].split(".")[0]
                    plt.savefig(outputfolder +filename +"_"+element+ "."+extension, dpi=300, format=extension)
                    if verbose:
                        plt.show()
                    plt.close()


if __name__ == "__main__":
    wordcloud()