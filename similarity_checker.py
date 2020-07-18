import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import csv


def readCouples():
    couples = []
    similarities = []
    with open('WordSim353\WordSim353.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile,skipinitialspace=True)
        next(csvreader)
        for row in csvreader:
            couples.append(row[0:2])
            similarities.append(float(row[2]))
    return couples,normalizeData(similarities)

def getMaxDepthWordnet():
    return max(max(len(hyp_path) for hyp_path in ss.hypernym_paths()) for ss in wn.all_synsets())

def getSinDepth(sin):
    maxdepth = sin.max_depth()
    if maxdepth == 0:
        return 1
    else:
        return maxdepth

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def wu_palmer(term_couples):
    similarities = []
    for term_couple in term_couples:
        max_sim = 0
        for first_sin in wn.synsets(term_couple[0]):
            first_sin_depth = getSinDepth(first_sin)
            for second_sin in wn.synsets(term_couple[1]):
                LCS = getSinDepth(first_sin.lowest_common_hypernyms(second_sin, simulate_root=True)[0])
                second_sin_depth = getSinDepth(second_sin)
                similarity = (2*LCS)/(first_sin_depth+second_sin_depth)
                if similarity > max_sim:
                    max_sim = similarity
        similarities.append(float(max_sim))
    return similarities

def shortest_path(term_couples):
    wordnet_maxdepth = getMaxDepthWordnet()
    similarities = []
    for term_couple in term_couples:
        max_sim = 0
        for first_sin in wn.synsets(term_couple[0]):
            for second_sin in wn.synsets(term_couple[1]):
                similarity = 2*wordnet_maxdepth - first_sin.shortest_path_distance(second_sin, simulate_root=True)
                if similarity > max_sim:
                    max_sim = similarity
        similarities.append(float(max_sim))
    #normalize because shortest_path range is 0-2
    return normalizeData(similarities)

def leack_chodorow(term_couples):
    wordnet_maxdepth = getMaxDepthWordnet()
    #print("WORDNET MAX DEPTH: ",wordnet_maxdepth)
    similarities = []
    for term_couple in term_couples:
        max_sim = -100000
        for first_sin in wn.synsets(term_couple[0]):
            for second_sin in wn.synsets(term_couple[1]):
                similarity = -np.log((first_sin.shortest_path_distance(second_sin, simulate_root=True)+1)/(2*wordnet_maxdepth+1))
                if similarity > max_sim:
                    max_sim = similarity
        similarities.append(float(max_sim))
    return normalizeData(similarities)

def main():
    #nltk.download('wordnet')
    print("################# Starting..")
    term_couples,similarities = readCouples()
    print("################# READ PHASE COMPLETED")
    #print(similarities)

    wu_palmer_sim = wu_palmer(term_couples)
    print("################# WU PALMER CALCULUS COMPLETED")
    
    #normalize because shortest_path range is 0-2
    short_path_sim = normalizeData(shortest_path(term_couples))
    print("################# SHORTEST PATH CALCULUS COMPLETED")

    leackchodorow_sim = leack_chodorow(term_couples)
    print("################# LEAKCOCK CHODOROW CALCULUS COMPLETED")

    print("WuPalmer_pearson: ",pearsonr(wu_palmer_sim,similarities))
    print("WuPalmer_spearman: ",spearmanr(wu_palmer_sim,similarities))
    print("ShortestPath_pearson: ",pearsonr(short_path_sim,similarities))
    print("ShortestPath_spearman: ",spearmanr(short_path_sim,similarities))
    print("LeakcockChodorow_pearson: ",pearsonr(leackchodorow_sim,similarities))
    print("LeakcockChodorow_spearman: ",spearmanr(leackchodorow_sim,similarities))

    #for debug purposes
    print(min(similarities),"--",max(similarities))
    print(min(wu_palmer_sim),"--",max(wu_palmer_sim))
    print(min(short_path_sim),"--",max(short_path_sim))
    print(min(leackchodorow_sim),"--",max(leackchodorow_sim))
    
if __name__ == "__main__":
    main()