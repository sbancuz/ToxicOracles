# simple script to report the best 10 results from the folder finalResults
import sys
import os
import json
import click
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolutionary import Archive, get_score

def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)

def higherScore(archive: Archive, quantity: int):
    # given an an archive, return the iterations with the highest score
    best = []
    for run in archive.runs:
        best.append(max(run.taken, key=lambda x: get_score(list(x.criterion.values()), "max")))
    
    # return the best iterations and the config
    return sorted(best, key=lambda x: get_score(list(x.criterion.values()), "max"), reverse=True)[:quantity], archive.config




@click.command()
@click_option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False),
    help="Folder containing the folders of archive files",
    default="results/finalTests/"
)
@click_option(
    "--quantity",
    "-q",
    type=int,
    default=10,
    help="Number of best results to show"
)
def printBest(input, quantity):
    # load all the folders in the input folder
    folders = [f.path for f in os.scandir(input) if f.is_dir()] #and not "vicunaUC_vicunaUC" in f.path]
    # load all the jsons in the folders
    archives = []

    best=[]
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".json") and not filename.startswith("baseline"):
                #print(f"Loading {folder+filename}")
                with open(os.path.join(folder, filename), "r") as file:
                    archives.append(Archive.from_dict(json.load(file)))
    
    for archive in archives:
        
        bestIterations, config = higherScore(archive, quantity)
        # append the best iterations to the best list, without considering the quantity
        best.append((bestIterations, config))
    
    
    # sort the best list by the score of the best iteration
    best.sort(key=lambda x: get_score(list(x[0][0].criterion.values()), "max"), reverse=True)
    best = best[:quantity]
    for i in best:
        print(f"Score: {get_score(list(i[0][0].criterion.values()), 'max')}")
        print(f"Config: {i[1]}")
        print(f"Iteration: {i[0][0]}")
        print("\n\n\n\n")


if __name__ == "__main__":
    printBest()




