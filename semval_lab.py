import hashlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr

SEMEVAL_ANNOTADED_FILE = "./semval_utils/it.test.data.annotated.tsv"
NASARI_PATH = "./semval_utils/mini_NASARI.tsv"
SENSES2SYNSETS_PATH = "semval_utils\SemEval17_IT_senses2synsets.txt"
SEMEVAL_ANNOTADED_FILE_CONSEGNA2 = "./semval_utils/it.test.data.annotated.consegna2.tsv"
#NASARI_PATH_TERMS = "./summarizerutils/dd-small-nasari-15.txt"
NASARI_PATH_TERMS = "./summarizerutils/dd-nasari.txt"

def get_range(surname):
    nof_elements = 500
    base_idx = (abs(int(hashlib.sha512(surname.encode('utf-8')).hexdigest(), 16)) % 10)
    idx_intervallo = base_idx * 50+1
    return idx_intervallo

#read the manually annotated file and return a dataframe
def readAnnotatedCouples():
    return pd.read_csv(SEMEVAL_ANNOTADED_FILE, sep='	', names=['first','second','score'])

def read_nasari():
    nasari_df = pd.read_csv(NASARI_PATH, sep='$', names=['babel'])#fake separator
    nasari_df[['babel','terms']] = nasari_df["babel"].str.split("	", 1, expand=True)
    nasari_df[['babel','lemma']] = nasari_df["babel"].str.split("__", 1, expand=True)
    nasari_df['lemma'] = nasari_df['lemma'].str.lower()
    return nasari_df

#estrae i termini(embed) derivanti da ogni babelid dell'array topic
#ritorna un array di array (uno per ogni babelid)
def getTermsFromBabelIds(topic, nasari_df):
    if len(topic)==0:
        return list()
    nasari_vect = []
    for single_topic in topic:
        nasari_terms = nasari_df.loc[nasari_df[nasari_df.columns[0]] == single_topic]['terms'].tolist()
        if len(nasari_terms) == 0:
            #nasari_vect.append([])
            continue
        else:
            nasari_vect.append(nasari_terms[0].split("	"))
    return nasari_vect

#return a dict <term,babelsynids>
def read_sense2synset():
    dict_to_ret = {}
    temp_synset_list = []
    last_term_seen = None
    first = True
    with open(SENSES2SYNSETS_PATH,encoding="utf-8") as f:
        while True:
            line = f.readline().strip('\n')
            if not line: 
                break
            if line.startswith('#'):#è un termine
                if first:
                    last_term_seen = line[1:]
                    first = False
                else:
                    dict_to_ret[last_term_seen] = temp_synset_list.copy()
                    temp_synset_list.clear()
                    last_term_seen = line[1:]
            else:#è un babelsynset
                temp_synset_list.append(line)

    return dict_to_ret

#retrieve babel synsets terms related to columns first and second of annotated_couples dataframe
def getBabelTerms(annotated_couples):
    babel_term_synset_mapper = read_sense2synset()
    nasari_df = read_nasari()
    annotated_couples['first_syn_terms_embed'] = None
    annotated_couples['second_syn_terms_embed'] = None
    for i in annotated_couples.index:
        term1 = annotated_couples.iloc[i, :]['first']
        if not term1 in babel_term_synset_mapper:
            continue
        first_syns = babel_term_synset_mapper[term1]
        first_syn_terms = getTermsFromBabelIds(first_syns,nasari_df)

        if len(first_syn_terms) == 0:
            continue

        term2 = annotated_couples.iloc[i, :]['second']
        if not term2 in babel_term_synset_mapper:
            continue
        second_syns = babel_term_synset_mapper[term2]
        second_syn_terms = getTermsFromBabelIds(second_syns,nasari_df)

        if len(second_syn_terms) == 0:
            continue

        annotated_couples.at[i, 'first_syn_terms_embed'] = first_syn_terms
        annotated_couples.at[i, 'second_syn_terms_embed'] = second_syn_terms

    return annotated_couples,babel_term_synset_mapper

def cosine_similarity(x, y):
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

#calculate max cosine similarity between first_syn_terms_embed and second_syn_terms_embed
#sens2syn_dict = a dict <term,babelsynids>
def calculateNasariSimilarity(babelTerms,sens2syn_dict):
    babelTerms['nasari_cosin_similarity'] = None
    babelTerms['most_similar_syn1'] = None
    babelTerms['most_similar_syn2'] = None
    for i in babelTerms.index:
        first_syn_terms_embed = babelTerms.iloc[i, :]['first_syn_terms_embed']
        second_syn_terms_embed = babelTerms.iloc[i, :]['second_syn_terms_embed']
        maxSim = -100
        idx_max_synset1 = 0 #indice del babelsynset di first che massimizza la similarità
        idx_max_synset2 = 0 #indice del babelsynset di second che massimizza la similarità
        if (first_syn_terms_embed is None) or (second_syn_terms_embed is None):
            continue

        tmp_idx1 = 0
        tmp_idx2 = 0
        for term in first_syn_terms_embed:
            tmp_idx2 = 0
            for term2 in second_syn_terms_embed:
                sim = cosine_similarity(term,term2)
                if sim > maxSim:
                    maxSim = sim
                    idx_max_synset1 = tmp_idx1
                    idx_max_synset2 = tmp_idx2
                tmp_idx2+=1
            tmp_idx1+=1

        babelTerms.at[i, 'nasari_cosin_similarity'] = maxSim
        babelTerms.at[i, 'most_similar_syn1'] = sens2syn_dict[babelTerms.iloc[i, :]['first']][idx_max_synset1]
        babelTerms.at[i, 'most_similar_syn2'] = sens2syn_dict[babelTerms.iloc[i, :]['second']][idx_max_synset2]

    return babelTerms

def printSpearmanPearson(list1, list2):

    #removing None similarity from list2
    res = [i for i in range(len(list2)) if list2[i] == None]
    print(res)
    for indexNone in sorted(res, reverse=True):
        del list1[indexNone]
        del list2[indexNone]

    print("pearson: ",pearsonr(list1,list2))
    print("spearman: ",spearmanr(list1,list2))

def consegna1():
    input_name = "Coluccia"

    values = []
    sx = get_range(input_name)
    values.append(sx)
    dx = sx+50-1
    intervallo = "" + str(sx) + "-" + str(dx)
    print('{:15}:\tcoppie nell\'intervallo {}'.format(input_name, intervallo))

    annotated_couples = readAnnotatedCouples()
    pd.to_numeric(annotated_couples['score'], errors='ignore')
    annotated_couples['score']=(annotated_couples['score']-annotated_couples['score'].min())/(annotated_couples['score'].max()-annotated_couples['score'].min())
    #print(annotated_couples)
    babelTerms,sens2syn_dict = getBabelTerms(annotated_couples)
    #print(babelTerms)
    nasari_sim = calculateNasariSimilarity(babelTerms,sens2syn_dict)
    print(nasari_sim)
    printSpearmanPearson(annotated_couples['score'].tolist(),nasari_sim['nasari_cosin_similarity'].tolist())
    #i coefficenti non evidenziano una forte correlazione --> secondo me perchè ci sono alcuni score molto distanti (soprattutto quelli che io ho messo a 0 o a 4)
    return nasari_sim #mi serve per la consegna2

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------

def readNasariDfTerms():
    NASARI_DF = pd.read_csv(NASARI_PATH_TERMS, sep='$', names=['babel'])#fake separator
    NASARI_DF[['babel','terms']] = NASARI_DF["babel"].str.split(";", 1, expand=True)
    NASARI_DF[['lemma','terms']] = NASARI_DF["terms"].str.split(";", 1, expand=True)
    NASARI_DF['lemma'] = NASARI_DF['lemma'].str.lower()
    return NASARI_DF

def getTermsFromBabelIds_consegna2(single_topic,nasaridf):
    nasari_terms = nasaridf.loc[nasaridf[nasaridf.columns[0]] == single_topic]['terms'].tolist()
    
    nasari_terms_filtered = []
    for term in nasari_terms:
        if term is None:
            continue
        words = term.split(';')
        #words.pop(0)#remove first
        for word in words:
            if word == "":
                continue
            splitted = word.split("_")
            if len(splitted) < 2:
                continue
            #print(splitted)
            nasari_terms_filtered.append(splitted[0])
    return nasari_terms_filtered

#it returns a dataframe with this structure: 'term1','term2','babel1','babel2','terms_in_bs1', 'terms_in_bs2'
def readSynsetManuallyAnnotated():
    df = pd.read_csv(SEMEVAL_ANNOTADED_FILE_CONSEGNA2, sep='	', names=['first','second','babel1','babel2'])
    df['terms_in_bs1'] = None
    df['terms_in_bs2'] = None

    nasari_df = readNasariDfTerms()

    for i in df.index:
        babel1 = df.iloc[i, :]['babel1']
        babel2 = df.iloc[i, :]['babel2']
        if babel1 is None:
            continue
        first_syn_terms = getTermsFromBabelIds_consegna2(babel1,nasari_df)

        if babel2 is None:
            continue
        second_syn_terms = getTermsFromBabelIds_consegna2(babel2,nasari_df)

        df.at[i, 'terms_in_bs1'] = first_syn_terms
        df.at[i, 'terms_in_bs2'] = second_syn_terms
    return df

def calculateBestSimilarityNasariSynset(annotated_df):
    #remove rows where at least one of the babel_terms list is empty
    filtered_df = annotated_df[(annotated_df.terms_in_bs1.map(len) > 0) & (annotated_df.terms_in_bs2.map(len) > 0)]
    filtered_df = filtered_df.reset_index()
    #print(filtered_df)
    #read nasari embed
    nasari_df = read_nasari()

    #read sense2synset
    #babel_term_synset_mapper = read_sense2synset()

    filtered_df['first_syn_terms_embed'] = None
    filtered_df['second_syn_terms_embed'] = None

    #calculate cosine similarity for each row
    for i in filtered_df.index:
        #get embedded
        '''
        babel1 = filtered_df.iloc[i, :]['babel1']
        babel2 = filtered_df.iloc[i, :]['babel2']
        babel1_embeds = getTermsFromBabelIds([babel1],nasari_df)[0]
        babel2_embeds = getTermsFromBabelIds([babel2],nasari_df)[0]
        filtered_df.at[i, 'first_syn_terms_embed'] = [babel1_embeds] #metto come lista perchè riuso il metodo cosinesimilarity che si aspetta una lista
        filtered_df.at[i, 'second_syn_terms_embed'] = [babel2_embeds]
        '''
        filtered_df,sens2syn_dict = getBabelTerms(filtered_df)
        nasari_sim_df = calculateNasariSimilarity(filtered_df,sens2syn_dict)
    #print(nasari_sim_df)
    #calculate accuracy over first babel
    most_similar_syn1 = nasari_sim_df['most_similar_syn1'].tolist()
    babel1 = nasari_sim_df['babel1'].tolist()
    correct = 0
    index = 0
    for nasari_babel in most_similar_syn1:
        my_babel = babel1[index]
        if nasari_babel == my_babel:
            correct+=1
        index+=1
    print("Accuracy over first babel: ",correct/len(most_similar_syn1))
    #calculate accuracy over second babel
    most_similar_syn2 = nasari_sim_df['most_similar_syn2'].tolist()
    babel2 = nasari_sim_df['babel2'].tolist()
    correct = 0
    index = 0
    for nasari_babel in most_similar_syn2:
        my_babel = babel2[index]
        if nasari_babel == my_babel:
            correct+=1
        index+=1
    print("Accuracy over second babel: ",correct/len(most_similar_syn2))
    #calculate accuracy over couple
    correct = 0
    index = 0
    for nasari_babel in most_similar_syn2:
        my_babel = babel2[index]
        my_babel1 = babel1[index]
        nasari_babel1 = most_similar_syn1[index]
        if nasari_babel == my_babel and nasari_babel1 == my_babel1:
            correct+=1
        index+=1
    print("Accuracy over couples: ",correct/len(most_similar_syn2))

def consegna2(consegna1_df):
    annotated_df = readSynsetManuallyAnnotated()
    print(annotated_df)
    calculateBestSimilarityNasariSynset(annotated_df)


def main():
    consegna1_df = consegna1()
    consegna2(consegna1_df)

if __name__ == "__main__":
    main()