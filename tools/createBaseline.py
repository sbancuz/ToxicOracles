import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolutionary import Archive, Config, Question, InitialQuestion
import click
import orjson



def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)




@click.command()
@click_option(
    "-i",
    "--input",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True),
    default="results/finalTests",
    help="The directory containing the folder with the final tests. (default: results/finalTests)",
)
@click_option(
    "-s",
    "--sample",
    type=click.Path(exists=False, resolve_path=False, dir_okay=False),
    default="max.json",
    help="The file containing the initial question. (default: max.json)",
)
def addBaseline(input, sample):
    # list only the folders in the input directory
    folders = [f for f in os.listdir(input) if os.path.isdir(os.path.join(input, f))]
    for folder in folders:
        # for each folder in the input directory, create a new archive with only the initial question, taken from the file sample
        # load the sample file if it exists
        with open(input+'/'+folder+"/"+sample, "r") as f:
            archive=Archive.from_dict(orjson.loads(f.read()))
            # drop all the iterations
            archive.config.iterations = 0
            for run in archive.runs:
                run.discarded=[]
                run.taken=[]
                run.start_time_timestamp=run.initial.start_time_response
                run.end_time_timestamp=run.initial.end_time_evaluation
                run.delta_time_timestamp=run.end_time_timestamp-run.start_time_timestamp
            archive.start_time_timestamp=archive.runs[0].start_time_timestamp
            archive.end_time_timestamp=archive.runs[-1].end_time_timestamp
            archive.delta_time_timestamp=archive.end_time_timestamp-archive.start_time_timestamp
            # save the new archive
            with open(input+'/'+folder+"/baseline.json", "w") as f:
                f.write(orjson.dumps(archive.to_dict(), option=orjson.OPT_INDENT_2).decode("utf-8"))
            


if __name__ == "__main__":
    addBaseline()


