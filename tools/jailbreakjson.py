# python script that loads a csv and creates a json with the following format, a list of objects made:
# id, number
# pattern: list of strings,
# name: string,
# text: string

import csv
import json



def main():
    with open('./examples/jailbreak-prompt.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=';', fieldnames=['id', 'pattern', 'name', 'created_at', 'text'])
        data = []
        for row in reader:
            # remove "[INSERT PROMPT HERE]" at the end of some prompts
            if row['text'].endswith('[INSERT PROMPT HERE]'):
                row['text'] = row['text'][:-len('[INSERT PROMPT HERE]')]

            
            data.append({
                'id': row['id'],
                'pattern': row['pattern'].split(','),
                'name': row['name'],
                'text': row['text']
            })
    with open('./examples/jailbreak-prompt.json', 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()

    