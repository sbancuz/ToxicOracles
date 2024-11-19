# given a directory, for each folder edit the json files that starts with ppl and ends with .json and add the sut field with the first part of the folder name, separated by an underscore. For example, if the folder is named "test1" and the file is named "ppl1.json", the sut field should be "test1".
#

import os
import json
import click
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolutionary import Archive, Config


def getPerplexityFiles(root):
    # for each folder in folder
    # get the perplexity files
    # return the list of files
    perplexityFiles = []
    # folders in the root folder
    folders = [os.path.join(root, folder) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder)) and folder != 'vicunaUC_vicunaUC']
    for folder in folders:
        # furthermore, search if there are folders named "perplexity" and add the files in those folders
        perplexityFolder = os.path.join(folder, 'perplexity')
        # if the folder exists
        if os.path.exists(perplexityFolder):
            # list the folders in perplexityFolder
            perpFolder= [os.path.join(perplexityFolder, folder) for folder in os.listdir(perplexityFolder) if os.path.isdir(os.path.join(perplexityFolder, folder))]
            for model in perpFolder:
                files = os.listdir(model)
                for file in files:
                    if file.startswith("ppl") and file.endswith(".json"):
                        perplexityFiles.append(os.path.join(model, file))
        else:
            # if the folder does not exist, list the files in the folder
            files = os.listdir(folder)
            for file in files:
                if file.startswith("ppl") and file.endswith(".json"):
                    perplexityFiles.append(os.path.join(folder, file))
    return perplexityFiles

def fixJsonFiles(data):
    #print(data)
    for file in data:
        print(file)
        with open(file, 'r') as f:
            data = json.load(f)
            # get the name of the grandparent folder
            name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))
            # handle field is the last part of the "results_file" field (last /), without the .json extension
            # an handle example is "max", "max-fs", etc
            # could be inferred from the configuration file, but it is easier to get it from the results path
            data['handle'] = os.path.basename(data['results_file']).split('.')[0]
            # load the corresponding json file, which is in the grandparent folder, named as the configuration

            relativeFile=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file))), data['handle']+'.json')
            # load the relative file and extract the sut and configuration from there
            with open(relativeFile, 'r') as f:
                relativeArchive = Archive.from_dict(json.load(f))
                data['config']=relativeArchive.config.to_dict()

            with open(relativeFile, 'r') as f:
                relativeData = json.load(f)
                # the files have the same number of runs
                for index in range(len(data['runs'])):
                    # print if the initial prompt are the same
                    if data['runs'][index]['initial']['prompt_from_dataset'] != relativeData['runs'][index]['initial']['prompt_from_dataset']:
                        print(f'{file} and {relativeFile} have different initial prompts')
                    else:
                        # add the score to the respetive run
                        data['runs'][index]['initial']['score'] = relativeData['runs'][index]['initial']['score']
                        for i, taken in enumerate(data['runs'][index]['taken']):
                            data['runs'][index]['taken'][i]['score'] = relativeData['runs'][index]['taken'][i]['score']
                        
                        



        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    
@click.command()
@click.option('--folder', '-f', help='Folder with the perplexity files', required=True, default='results/finalTests/')
def main(folder):
    perplexityFiles = getPerplexityFiles(folder)
    fixJsonFiles(perplexityFiles)

if __name__ == '__main__':
    main()