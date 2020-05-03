from py_babelnet.calls import BabelnetAPI
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

#indica di quanto vorrei ridurre la lunghezza del testo
REDUCTION_RATE = 0.5
TOPIC_METHOD = 'title'
babel_api = BabelnetAPI('07825ba5-007d-44fb-8d36-b70fcd096ef5')
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))
NASARI_PATH = "./summarizerutils/dd-small-nasari-15.txt"
#NASARI_PATH = "./summarizerutils/dd-nasari.txt"
NASARI_DF = None
BABEL_WEB_SERVICE = False #if true it makes an http request for each term to babel WS

def split_in_paragraph(doc):
    with open(doc,encoding="utf-8") as f:
        paragraphs = f.readlines()

    paragraphs = [x.strip() for x in paragraphs]
    paragraphs_filtered = []
    for paragraph in paragraphs:
        if not len(paragraph) == 0:
            paragraphs_filtered.append(paragraph)
    length = len(''.join(paragraphs_filtered))
    return paragraphs_filtered,length

#ritorna i babelsynset id rappresentanti il lemma passato in input
#se deep_in_terms==True--> cerca il lemma anche tra i termini
def getBabelSynset(lemma,deep_in_terms = False):

    if BABEL_WEB_SERVICE:
        #this version use an API call to BabelNet web service. It is very slow.
        senses = babel_api.get_senses(lemma = lemma, searchLang = "EN")
    else:
        senses = NASARI_DF.loc[NASARI_DF['lemma'] == lemma.lower()]['babel'].tolist()
        if deep_in_terms:
            senses.extend(NASARI_DF.loc[NASARI_DF['terms'].str.contains(lemma)]['babel'].tolist())

    return senses

#extract babelnet vectors id from topic
def getTopic(paragraphs,deep_in_terms=False):
    bab_synsets = []
    if TOPIC_METHOD == 'title':
        #the title is inside the link in the first paragraph
        last_word = paragraphs[0].replace('.html', '').split('/')[-1]
        if "-" not in last_word: 
            title_words = last_word.split('_')
        else:
            title_words = last_word.split('-')

        filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in title_words if (not w in STOP_WORDS) and w.isalpha()]
        for word in filtered_words:
            synsets = getBabelSynset(word,deep_in_terms)
            if synsets!=None and len(synsets)>0:
                if BABEL_WEB_SERVICE:
                    bab_synsets.append(synsets[0]['properties']['synsetID']['id'])#prendo il primo babelsynset
                else:
                    bab_synsets.append(synsets[0])
        return bab_synsets

#estrae i termini (con il loro peso) derivanti da ogni babelid dell'array topic
#ritorna anche i vectors nasari
def getTermsFromBabelIds(topic):
    if len(topic)==0:
        return list()
    nasari_vect = []
    for single_topic in topic:
        nasari_terms = NASARI_DF.loc[NASARI_DF[NASARI_DF.columns[0]] == single_topic]['terms'].tolist()
        
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
        nasari_vect.append(nasari_terms_filtered)
    return nasari_vect


#estrae i babel synset id da un paragrafo
def getBabelIdsFromParagraph(paragraph):
    bab_synsets = []
    words = paragraph.split()
    filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in words if (not w in STOP_WORDS) and w.isalpha()]
    for word in filtered_words:
        synsets = getBabelSynset(word)
        if synsets!=None and len(synsets)>0:
            if BABEL_WEB_SERVICE:
                bab_synsets.append(synsets[0]['properties']['synsetID']['id'])#prendo il primo babelsynset
            else:
                bab_synsets.append(synsets[0])
    return bab_synsets

#return a score for the paragraph_terms_vect. It get the max WO of each paragraph with context
def getScoreForParagraph(paragraph_terms_vect, context_terms_vect):
    #calculate square-rooted Weighted Overlap between parameters
    '''
    overlap_terms = set(paragraph_terms).intersection(set(context_terms))
    if len(overlap_terms) == 0:
        return 0
    numerator = 0
    for term in overlap_terms:
        numerator+= paragraph_terms.index(term)+context_terms.index(term)
    numerator = numerator**-1
    denominator = (2*len(overlap_terms))**-1
    return numerator/denominator
    '''
    max = 0
    for paragraph_terms in paragraph_terms_vect:
        for context_terms in context_terms_vect:
            overlap_terms = set(paragraph_terms).intersection(set(context_terms))
            if len(overlap_terms) == 0:
                continue
            numerator = 0
            for term in overlap_terms:
                numerator+= paragraph_terms.index(term)+context_terms.index(term)
            if numerator == 0:
                continue
            numerator = numerator**-1
            denominator = (2*len(overlap_terms))**-1
            result = numerator/denominator
            if result > max:
                max = result
    return max

def filter_paragraphs(paragraphs, context_nas_vect, reduction_rate):
    #get the first summary_lenght_rate best paragraphs
    score_dict = {} #il value della chiave i-esima è lo score del paragrafo i-esimo
    paragraphs.pop(0)
    pos = 0
    for paragraph in paragraphs:
        babelIds = getBabelIdsFromParagraph(paragraph)
        paragraph_nas_vect = getTermsFromBabelIds(babelIds)
        score_dict[pos] = getScoreForParagraph(paragraph_nas_vect,context_nas_vect)
        pos+=1

    num_of_paragraphs_to_get = (len(score_dict)*(1-reduction_rate))
    print("Extracting the best ",num_of_paragraphs_to_get)
    score_dict_ordered = {k: v for k, v in sorted(score_dict.items(), reverse=True, key=lambda item: item[1])}

    paragraphs_to_extract = []
    scores_to_ret = []
    taken = 0
    for paragraph_pos in score_dict_ordered.keys():
        paragraphs_to_extract.append(paragraphs[paragraph_pos])
        scores_to_ret.append(score_dict_ordered[paragraph_pos])
        taken+=1
        if taken >= num_of_paragraphs_to_get:
            return paragraphs_to_extract,scores_to_ret

    return paragraphs_to_extract,scores_to_ret

def summarize_doc(doc, deep=True):
    paragraphs, doc_len = split_in_paragraph(doc)
    #print(paragraphs," ----- ",doc_len)
    topic = getTopic(paragraphs,deep)
    print("TOPICS: ",topic)
    context_nas_vect = getTermsFromBabelIds(topic)
    print("CONTEXT: ",context_nas_vect)
    salient_paragraphs, scores = filter_paragraphs(paragraphs,context_nas_vect,REDUCTION_RATE)
    print("@@@@@@@@@@@@@@@@@@@@@@ SUMMARY @@@@@@@@@@@@@@@@@@@@")
    i = 0
    for salient_paragraph in salient_paragraphs:
        print("--> ",scores[i],"= ",salient_paragraph)
        i+=1
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")

def main():
    global NASARI_DF
    doc_paths = ['./summarizerutils/text-documents/Andy-Warhol.txt','./summarizerutils/text-documents/Ebola-virus-disease.txt','./summarizerutils/text-documents/Life-indoors.txt','./summarizerutils/text-documents/Napoleon-wiki.txt']
    NASARI_DF = pd.read_csv(NASARI_PATH, sep='$', names=['babel'])#fake separator
    NASARI_DF[['babel','terms']] = NASARI_DF["babel"].str.split(";", 1, expand=True)
    NASARI_DF[['lemma','terms']] = NASARI_DF["terms"].str.split(";", 1, expand=True)
    NASARI_DF['lemma'] = NASARI_DF['lemma'].str.lower()
    #print(NASARI_DF.head())

    for doc in doc_paths:
        print("Processing doc: ",doc)
        if "Napoleon" in doc:
            summarize_doc(doc, True)
        else:
            summarize_doc(doc, True)

if __name__ == "__main__":
    main()