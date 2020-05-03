import nltk
import random
from random import randint
from random import seed
from nltk.corpus import framenet as fn
from nltk.corpus.reader.framenet import PrettyList
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
import hashlib

from operator import itemgetter
import pprint


STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

GOLD_STANDARD_FRAME_NAME_1 = 'accuracy.n.01'
GOLD_STANDARD_LU_1 = {
    'accuracy': 'accuracy.n.01',
    'accurate': 'accurate.a.01',
    'accurately': 'accurately.r.02',
    'exact': 'exact.a.01',
    'inaccuracy': 'inaccuracy.n.01',
    'inaccurate': 'inaccurate.a.01',
    'inaccurately': 'inaccurately.r.01',
    'off': 'off.a.01',
    'on': 'on.a.01',
    'precise': 'precise.a.01',
    'precision': 'preciseness.n.02',
    'true': 'true.a.02'
}
GOLD_STANDARD_FRAME_ELEM_1 = {
    'Agent': 'agent.n.01',
    'Circumstances': 'circumstance.n.01',
    'Degree': 'degree.n.01',
    'Deviation': 'deviation.n.01',
    'Domain': 'domain.n.03',
    'Frequency': 'frequency.n.02',
    'Instrument': 'instrumental_role.n.01',
    'Means': 'mean.n.01',
    'Outcome': 'result.n.03',
    'Place': 'rate.v.01',
    'Target': 'aim.n.02',
    'Target_value': 'value.n.01',
    'Time': 'time.n.01'
}
#######################################################
GOLD_STANDARD_FRAME_NAME_2 = 'noise.n.03'
GOLD_STANDARD_LU_2 = {
    'babble': 'babble.v.01',
    'bark': 'bark.n.02',
    'bawl': 'yawp.v.01',
    'bellow': 'bawl.v.01',
    'bleat': 'bleat.v.01',
    'bray': 'bray.v.03',
    'burble': 'burble.v.01',
    'cackle': 'yak.n.01',
    'chirp': 'chirp.n.01',
    'chirrup': 'peep.v.03',
    'chuckle': 'chortle.n.01',
    'cluck': 'cluck.v.01',
    'coo': 'coo.v.01',
    'croak': 'croak.v.02',
    'croon': 'croon.v.01',
    'crow': 'crow.v.03',
    'cry': 'cry.v.01',
    'drone': 'drone.v.01',
    'gasp': 'pant.v.01',
    'grate': 'grate.n.02',
    'groan': 'groan.n.01',
    'growl': 'grumble.v.03 ',
    'grunt': 'grunt.n.01',
    'gurgle': 'gurgle.v.02',
    'hiss': 'hiss.v.01',
    'hoot': 'hoot.n.01',
    'howl': 'roar.v.01',
    'moan': 'groan.n.01',
    'murmur': 'mutter.n.01',
    'purr': 'purr.n.01',
    'rap': 'pat.n.01',
    'rasp': 'rasp.n.01',
    'rattle': 'rattle.n.01',
    'roar': 'boom.n.01',
    'rumble': 'rumble.n.01',
    'scream': 'scream.n.01',
    'screech': 'screech.n.01',
    'shriek': 'scream.n.01',
    'shrill': 'shriek.v.01',
    'snarl': 'snarl.n.01',
    'snort': 'boo.n.01',
    'splutter': 'spatter.n.01',
    'sputter': 'spatter.n.01',
    'squawk': 'squawk.v.01',
    'squeak': 'whine.v.03',
    'squeal': 'squeal.v.01',
    'thunder': 'boom.n.01',
    'titter': 'titter.n.01',
    'trill': 'trill.n.02',
    'trumpet': 'trumpet.v.03',
    'twitter': 'chitter.v.01',
    'wail': 'howl.v.01',
    'warble': 'warble.v.01',
    'wheeze': 'wheeze.n.01',
    'whimper': 'wail.v.02',
    'whine': 'snivel.v.01',
    'whoop': 'whoop.n.01',
    'yell': 'cry.n.01',
    'yelp': 'yelp.v.01'
}
GOLD_STANDARD_FRAME_ELEM_2 = {
    'Addressee': 'addressee.n.01',
    'Back': 'back.n.03',
    'Degree': 'degree.n.01',
    'Depictive': 'delineative.s.01',
    'Explanation': 'explanation.n.01',
    'Internal_cause': 'cause.n.01',
    'Manner': 'manner.n.01',
    'Means': 'means.n.01',
    'Medium': 'medium.n.01',
    'Message': 'message.n.02',
    'Place': 'topographic_point.n.01',
    'Speaker': 'speaker.n.01',
    'Time': 'time.n.05',
    'Topic': 'subject.n.01',
    'Voice': 'spokesperson.n.01'
}
#####################################################
GOLD_STANDARD_FRAME_NAME_3 = 'vehicle.n.01'
GOLD_STANDARD_LU_3 = {
}
GOLD_STANDARD_FRAME_ELEM_3 = {
    'Area': 'area.n.01',
    'Cotheme': None,
    'Distance': 'distance.n.01',
    'Driver': 'driver.n.01',
    'Duration': 'duration.n.01',
    'Goal': 'goal.n.02',
    'Manner': 'manner.n.01',
    'Path': 'path.n.04',
    'Road': 'road.n.01',
    'Route': 'road.n.01',
    'Source': 'beginning.n.04',
    'Speed': 'speed.n.01',
    'Theme': 'subject.n.01',
    'Time': 'time.n.04',
    'Vehicle': 'vehicle.n.01'
}
#######################################################
GOLD_STANDARD_FRAME_NAME_4 = 'relevant.a.01'
GOLD_STANDARD_LU_4 = {
    'irrelevant': 'irrelevant.a.01',
    'pertinent': 'pertinent.s.01',
    'play (into)': None,
    'relevant': 'relevant.a.01'
}
GOLD_STANDARD_FRAME_ELEM_4 = {
    'Cognizer': None,
    'Degree': 'degree.n.01',
    'Endeavor': 'endeavor.v.01',
    'Phenomenon': 'phenomenon.n.01',
    'Specification': 'specification.n.01'
}
#######################################################
GOLD_STANDARD_FRAME_NAME_5 = 'giving.n.01'
GOLD_STANDARD_LU_5 = {
    'advance': 'overture.n.03',
    'bequeath': 'bequeath.v.01',
    'bequest': 'bequest.n.01',
    'charity': 'charity.n.03',
    'confer (upon)': None,
    'contribute': 'contribute.v.2',
    'contribution': 'contribution.n.02',
    'donate': 'donate.v.01',
    'donation': 'contribution.n.02',
    'donor': 'donor.n.01',
    'endow': 'endow.v.02',
    'fob off': None,
    'foist': 'foist.v.01',
    'gift': 'gift.n.01',
    'give': 'give.v.03',
    'give out': None,
    'hand': 'hand.n.12',
    'hand in': None,
    'hand out': None,
    'hand over': None,
    'leave': 'leave.v.01',
    'pass out': None,
    'treat': 'process.v.01',
    'volunteer': 'volunteer.n.02',
    'will': 'will.n.03'
}
GOLD_STANDARD_FRAME_ELEM_5 = {
    'Circumstances': 'circumstance.n.01',
    'Depictive': 'delineative.s.01',
    'Donor': 'donor.n.01',
    'Explanation': 'explanation.n.01',
    'Imposed_purpose': 'determination.n.02',
    'Manner': 'manner.n.01',
    'Means': 'mean.n.01',
    'Period_of_iterations': 'iteration.n.01',
    'Place': 'topographic_point.n.01',
    'Purpose': 'purpose.n.01',
    'Recipient': 'recipient.n.01',
    'Theme': 'subject.n.01',
    'Time': 'time.n.05'
}


def print_frames_with_IDs():
    for x in fn.frames():
        print('{}\t{}'.format(x.ID, x.name))

def get_frams_IDs():
    return [f.ID for f in fn.frames()]   

def getFrameSetForStudent(surname, list_len=5):
    nof_frames = len(fn.frames())
    base_idx = (abs(int(hashlib.sha512(surname.encode('utf-8')).hexdigest(), 16)) % nof_frames)
    framenet_IDs = get_frams_IDs()
    i = 0
    offset = 0 
    seed(1)
    while i < list_len:
        fID = framenet_IDs[(base_idx+offset)%nof_frames]
        f = fn.frame(fID)
        fNAME = f.name
        print('\tID: {a:4d}\tframe: {framename}'.format(a=fID, framename=fNAME))
        offset = randint(0, nof_frames)
        i += 1

def extract_sense_context(sense):
    examples = set()
    for sentence in sense.examples():
        examples.update(word_tokenize(sentence))
    
    for hyp in sense.hyponyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)
    
    for hyp in sense.hypernyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)

    examples.update(word_tokenize(sense.definition()))
    filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in examples if (not w in STOP_WORDS) and w.isalpha()]
    return filtered_words

def computeoverlap(signature, context):
    return len(set(signature).intersection(set(context)))


def extract_frame_element_context(frame_element):
    frame_elem_context = set()

    #processing frame definition
    filtered_definition = [LEMMATIZER.lemmatize(w.lower()) for w in frame_element.definition if (not w in STOP_WORDS) and w.isalpha()]
    for word in filtered_definition:
        frame_elem_context.update(word)

    return frame_elem_context

def extract_frame_context(frame_id):
    frame = fn.frame(frame_id)
    frame_context = set()

    #processing frame definition
    filtered_definition = [LEMMATIZER.lemmatize(w.lower()) for w in frame.definition if (not w in STOP_WORDS) and w.isalpha()]
    for word in filtered_definition:
        frame_context.update(word)

    #processing frame elements definition
    for frame_elem in frame.FE:
        filtered_definition = [LEMMATIZER.lemmatize(w.lower()) for w in frame.FE[frame_elem].definition if (not w in STOP_WORDS) and w.isalpha()]
        for word in filtered_definition:
            frame_context.update(word)

    return frame_context

def assignSinsetToFrameName(frame_name,frame_context):
    best_sense = None
    max_overlap = -1
    if len(frame_name.split("_")) > 1:
        frame_name = frame_name.split("_")[-1]
    frame_name = LEMMATIZER.lemmatize(frame_name.lower())
    for sense in wn.synsets(frame_name):
        sense_context = extract_sense_context(sense)
        overlap = computeoverlap(sense_context,frame_context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    if best_sense is None:
        return best_sense         
    return best_sense.name()

def assignSinsetToLexicalUnits(frame_id,frame_context):
    sinsets = dict()
    frame = fn.frame(frame_id)
    for lexical_unit in frame.lexUnit:
        lexical_unit = lexical_unit.split(".")[0]
        lexical_unit = LEMMATIZER.lemmatize(lexical_unit.lower())
        sinsets[lexical_unit] = assignSinsetToFrameName(lexical_unit,frame_context)
    return sinsets

def assignSinsetToFrameElements(frame_id,frame_context):
    sinsets = dict()
    frame = fn.frame(frame_id)
    for frame_elem in frame.FE:
        sinsets[frame.FE[frame_elem].name] = assignSinsetToFrameName(frame.FE[frame_elem].name,extract_frame_element_context(frame.FE[frame_elem]))
    return sinsets

def processFrame(frame_name,frame_id):
    print("######### Processing frame id: ", frame_id)
    frame_context = extract_frame_context(frame_id)
    synset_frame_name = assignSinsetToFrameName(frame_name,frame_context)
    synset_lexical_units = assignSinsetToLexicalUnits(frame_id,frame_context)
    synset_frame_elements = assignSinsetToFrameElements(frame_id,frame_context)
    return synset_frame_name, synset_lexical_units, synset_frame_elements

def computeAccuracy(frame_index,synset_frame_name,synset_lexical_units,synset_frame_elements):
    total = 1 + len(synset_frame_name) + len(synset_lexical_units) + len(synset_frame_elements)
    correct = 0
    if frame_index == 1:
        if synset_frame_name == GOLD_STANDARD_FRAME_NAME_1:
            correct+=1
        for sin_lu in synset_lexical_units:
            if synset_lexical_units[sin_lu] == GOLD_STANDARD_LU_1[sin_lu]:
                correct+=1
        for sin_frame_elem in synset_frame_elements:
            if synset_frame_elements[sin_frame_elem] == GOLD_STANDARD_FRAME_ELEM_1[sin_frame_elem]:
                correct+=1
    if frame_index == 2:
        if synset_frame_name == GOLD_STANDARD_FRAME_NAME_2:
            correct+=1
        for sin_lu in synset_lexical_units:
            if synset_lexical_units[sin_lu] == GOLD_STANDARD_LU_2[sin_lu]:
                correct+=1
        for sin_frame_elem in synset_frame_elements:
            if synset_frame_elements[sin_frame_elem] == GOLD_STANDARD_FRAME_ELEM_2[sin_frame_elem]:
                correct+=1
    if frame_index == 3:
        if synset_frame_name == GOLD_STANDARD_FRAME_NAME_3:
            correct+=1
        for sin_lu in synset_lexical_units:
            if synset_lexical_units[sin_lu] == GOLD_STANDARD_LU_3[sin_lu]:
                correct+=1
        for sin_frame_elem in synset_frame_elements:
            if synset_frame_elements[sin_frame_elem] == GOLD_STANDARD_FRAME_ELEM_3[sin_frame_elem]:
                correct+=1
    if frame_index == 4:
        if synset_frame_name == GOLD_STANDARD_FRAME_NAME_4:
            correct+=1
        for sin_lu in synset_lexical_units:
            if synset_lexical_units[sin_lu] == GOLD_STANDARD_LU_4[sin_lu]:
                correct+=1
        for sin_frame_elem in synset_frame_elements:
            if synset_frame_elements[sin_frame_elem] == GOLD_STANDARD_FRAME_ELEM_4[sin_frame_elem]:
                correct+=1
    if frame_index == 5:
        if synset_frame_name == GOLD_STANDARD_FRAME_NAME_5:
            correct+=1
        for sin_lu in synset_lexical_units:
            if synset_lexical_units[sin_lu] == GOLD_STANDARD_LU_5[sin_lu]:
                correct+=1
        for sin_frame_elem in synset_frame_elements:
            if synset_frame_elements[sin_frame_elem] == GOLD_STANDARD_FRAME_ELEM_5[sin_frame_elem]:
                correct+=1
    
    return correct/total

def main():
    getFrameSetForStudent('Coluccia')
    frames = {'accuracy':1903,'noise':39,'vehicle':1690,'relevant':2530,'giving':139} #frame_name:frame_id
    pp = pprint.PrettyPrinter(indent=4)

    frame_index = 1
    for frame_name in frames:
        synset_frame_name, synset_lexical_units, synset_frame_elements = processFrame(frame_name,frames[frame_name])
        print("\t---SYNSET FOR FRAME NAME: ",synset_frame_name)
        print("\t---SYNSET FOR LEXICAL UNITS: ")
        pp.pprint(synset_lexical_units)
        print("\t---SYNSET FOR FRAME ELEMENTS: ")
        pp.pprint(synset_frame_elements)
        print("\t---ACCURACY: ",computeAccuracy(frame_index,synset_frame_name,synset_lexical_units,synset_frame_elements))
        frame_index += 1
        print("#########\n")

if __name__ == "__main__":
    #nltk.download('framenet_v17')
    main()


'''
student: coluccia
        ID: 1903        frame: Accuracy
        ID:   39        frame: Communication_noise
        ID: 1690        frame: Use_vehicle
        ID: 2530        frame: Being_relevant
        ID:  139        frame: Giving
'''