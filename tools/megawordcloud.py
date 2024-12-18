# mega wordcloud
import sys
import os
import json
import click
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image

from evolutionary import Archive

def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

@click.command()
@click_option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False),
    help="Folder containing the folders of archive files",
    default="results/finalTests/"
)
def load(input):
    # load all the folders in the input folder
    folders = [f.path for f in os.scandir(input) if f.is_dir()] #and not "vicunaUC_vicunaUC" in f.path]
    # load all the jsons in the folders
    generatedPrompts = []

    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".json") and not filename.startswith("Jailbreak"):
                with open(os.path.join(folder, filename), "r") as file:
                    archive=Archive.from_dict(json.load(file))
                    if len(archive.runs)>0:
                        for run in archive.runs:
                            generatedPrompts.append(run.initial.prompt_from_dataset+'\n')    
                            for taken in run.taken:
                                generatedPrompts.append(taken.generated_prompt_for_sut+'\n')
    
    print("Loaded all prompts "+str(len(generatedPrompts)))

    # read the mask
    masks = []
    rootdir = "tools/mask/"
    #masks = os.listdir(rootdir)
    masks=["mask.7.png"]

    for mask in masks:
        if mask.endswith(".png"):
            maskArray = np.array(Image.open(rootdir+mask))
            wc = WordCloud(
                width=4961,
                height=3508,
                background_color='white',
                stopwords=STOPWORDS,
                min_font_size=5,
                max_words=100000000000,
                mask=maskArray,
                font_path="tools/fonts/cmunrm.ttf",
                color_func=lambda *args, **kwargs: (91,120,164),
                max_font_size=25
            )
            # Generate a word cloud
            wc.generate("".join(generatedPrompts))
            # save the svg
            image=wc.to_svg(embed_font=True, embed_image=False)
            # replace "fill:(91, 120, 164)" with "fill:rgb(91,120,164)"
            image = image.replace("fill:(91, 120, 164)", "fill:rgb(91,120,164)")
            phrase='To Vincenzo, the best Research Director one could ever ask for. Thank you for all your support and guidance, Simone'
            # add to svg the text in the bottom right corner, which is the highest point of the wordcloud
            image = image.replace("</svg>", '<text  x="1670" y="1168" font-family="cmunrm" font-size="12" fill="rgb(49, 66, 92)" text-anchor="end">'+phrase+'</text></svg>')


            maskname = mask.split(".")[1]
            with open("wordcloud_"+maskname+".svg", "w") as file:
                file.write(image)
                
            # save the png
            maskname = mask.split(".")[1]
            wc.to_file("wordcloud_"+maskname+".png")
            print("Saved wordcloud_"+maskname+".png")

            # save the pdf
            maskname = mask.split(".")[1]
            wc.to_file("wordcloud_"+maskname+".pdf")




if __name__ == "__main__":
    load()




