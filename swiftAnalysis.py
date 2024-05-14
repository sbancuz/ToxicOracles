import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plotIterations(iterations, input):
    '''this function plots the cosine similarity for each iteration in the input file
    iterations: the list of dictionaries containing the cosine similarity for each iteration
    input: the input file, to save the plot in the same folder
    '''

    # create a dataframe with the iterations
    df = pd.DataFrame(iterations)
    directory=os.path.dirname(input)
    filename=os.path.basename(input)
    # create a boxplot for the iterations
    plt.figure(figsize=(10, 9))
    #ax = sns.boxplot(x="iteration", y="cosine similarity", data=df)
    ax = sns.violinplot(x="iteration", y="cosine similarity", data=df)
    ax.set_title("Cosine similarity between sentences, for "+filename)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("cosine similarity")
    # save the plot in the folder of the file
    #remove the extension
    filename = os.path.splitext(filename)[0]
    plt.savefig(f"{directory}/progressive/{filename}_iterations_similarity.png", dpi=300)
    #plt.show()

def main2():
    # load the json files in the current folder
    filepath = 'results/100/similarity results/'
    files = [filepath + f for f in os.listdir(filepath) if f.endswith('.json')]
    for f in files:
        #load the json file
        with open(f, 'r') as file:
            data = json.load(file)
            #plot the cosine similarity for each iteration
        plotIterations(data, f)
    
def main():
    # load all the files in the similarity results folder in a single list
    filepath = 'results/100/similarity results/'
    files = [filepath + f for f in os.listdir(filepath) if f.endswith('.json')]
    data = []
    for f in files:
        #load the json file
        with open(f, 'r') as file:
            for obj in json.load(file):
                data.append(obj)
    
    #plot the cosine similarity for each iteration
    plotIterations(data, 'results/100/similarity results/all.json')

    

if __name__ == '__main__':
    main()



