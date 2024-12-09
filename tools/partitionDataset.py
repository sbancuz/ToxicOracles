import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolutionary import load_questions
import click


def click_option(*args, **kwargs):
    if "show_default" not in kwargs:
        kwargs.update({"show_default": True})
    return click.option(*args, **kwargs)


@click.command()
@click_option(
    "-s",
    "--source",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
    default="dataset/questions",
    required=True,
    help="JSON File to load the questions from",
)
@click_option(
    "-q",
    "--questions",
    type=int,
    default=2,
    help="Number of questions to load from the dataset",
)
@click_option(
    "-o",
    "--output",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True),
    default="dataset/reduced",
    required=False,
    help="Path to save the reduced dataset to",
)
def main(source, questions, output):
    reduced = load_questions(file_path=source, amount=questions)
    # create document with the reduced questions
    with open(output + "/questions_reduced" + str(questions), "w") as f:
        for q in reduced:
            f.write(q)

    # write a json file with the questions
    # with open(output+"/questions_reduced"+str(questions)+".json", 'w') as f:
    #    json.dump(reduced, f)


if __name__ == "__main__":
    main()
