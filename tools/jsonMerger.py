import json
import os
import click

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolutionary import Archive


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


@click.command()
@click_option(
    "-o",
    "--output-path",
    type=click.Path(exists=False, resolve_path=True, dir_okay=False),
    help="Path to save the output",
    default="out/merged.json",
)
@click_option(
    "-f1",
    "--file1",
    type=click.Path(exists=False, resolve_path=True, dir_okay=False),
    help="Path to source file 1",
)
@click_option(
    "-f2",
    "--file2",
    type=click.Path(exists=False, resolve_path=True, dir_okay=False),
    help="Path to source file 2",
)
def run(output_path, file1, file2):
    # Open the first JSON file
    with open(file1, "r") as file:
        data1 = Archive.from_dict(json.loads(file.read()))

    # Open the second JSON file
    with open(file2, "r") as file:
        data2 = Archive.from_dict(json.loads(file.read()))

    if output_path is None:
        if not os.path.exists("out"):
            os.makedirs("out")

        i = 0
        while os.path.exists(f"out/output_{i}.json"):
            i += 1
        output_path = f"out/output_{i}.json"
    # If the number of iterations is different, the file with the least number of iterations will have its archive arrays (prompt, response and score) padded with the last element until the number of iterations is the same
    if data1.config.iterations != data2.config.iterations:
        if data1.config.iterations > data2.config.iterations:
            long_runs = data1
            short_runs = data2
        else:
            long_runs = data2
            short_runs = data1

        diff = long_runs.config.iterations - short_runs.config.iterations
        for run in long_runs.runs:
            run.taken = run.taken[:-diff]
            run.discarded = run.discarded[:-diff]
    else:
        long_runs = data1
        short_runs = data2

    long_runs.runs.extend(short_runs.runs)
    with open(output_path, "w") as f:
        f.write(long_runs.to_json())


if __name__ == "__main__":
    run()

