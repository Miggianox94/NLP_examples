import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
from random import sample
import random
from random import seed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def sample_corpus():
    corpus = semcor.tagged_sents(tag='both')[:200]
    sequence = [i for i in range(200)]
    # select a subset without replacement
    subset = sample(sequence, 50)
    sampled = []
    for index in subset:
        sampled.append(corpus[index])
    return sampled

def sample_noun(sentence):
    for element in sentence:
        if not isinstance(element[0], str) and element[0].label() == 'NN':
            return element.label().name(), element.label().synset()

def computecontext(sentence):
    filtered_sentence = [LEMMATIZER.lemmatize(w.label().name().lower()) for w in sentence if (not isinstance(w.label(),str)) and (not isinstance(w[0], str)) and (not w.label().name() in STOP_WORDS) and w.label().name().isalpha()]
    return filtered_sentence


def extract_signature(sense):
    examples = set()
    for sentence in sense.examples():
        examples.update(word_tokenize(sentence))

    examples.update(word_tokenize(sense.definition()))
    filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in examples if (not w in STOP_WORDS) and w.isalpha()]
    return filtered_words

def computeoverlap(signature, context):
    return len(set(signature).intersection(set(context)))

def leskAlgo(word,sentence):
    best_sense = None
    context = computecontext(sentence)
    max_overlap = -1
    for sense in wn.synsets(word):
        signature = extract_signature(sense)
        overlap = computeoverlap(signature,context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense

def main():
    #nltk.download('semcor')
    #nltk.download('stopwords')
    #nltk.download('punkt')
    seed(1)
    sentences = sample_corpus()
    correctCounter = 0
    correctCounterDummyAlgo = 0
    for sentence in sentences:
        word,gold_sense = sample_noun(sentence)
        best_sense = leskAlgo(word,sentence)
        if best_sense == gold_sense:
            correctCounter+=1
        if wn.synsets(word)[0] == gold_sense:
            correctCounterDummyAlgo+=1

    print("Accuracy: ",correctCounter/len(sentences))
    print("Accuracy with dummy algo: ",correctCounterDummyAlgo/len(sentences))

if __name__ == "__main__":
    main()