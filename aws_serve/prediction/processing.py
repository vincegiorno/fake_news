import numpy as np
import spacy
import en_core_web_sm
import re
from unidecode import unidecode
from spacy_langdetect import LanguageDetector

nlp = en_core_web_sm.load()
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

max_words = 30
max_sentences = 30
max_vocab = 150000
attention_dim = 100

class Cleaner(dict):
    """ Multiple-string-substitution dict """
    def _make_regex(self):
        """ Build re object based on the keys of the dictionary it is instantiated with"""
        return re.compile("|".join(map(re.escape, self.keys())))

    def __call__(self, match):
        """ Handler invoked for each regex match """
        return self[match.group(0)]

    def clean(self, text):
        """ Substitutes with value for each key and returns the modified text. """
        return self._make_regex().sub(self, text)

replacements = {"\n": " ",
                "\t": " ",
                "-": " ",
                "...": " ",
                "won't": "will not",
                "can't": "can not",
                "&": " and ",
                "\$*": "$",
                "Loading...": " ",
                "Continued...": " ",
                "\N{COPYRIGHT SIGN}": " ",
                "\N{NO-BREAK SPACE}": " ",
                "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}": " ",
                "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}": " ",
                '."': '".',
                '?"': '"?',
                '!"': '"!'
               }

entities = {'PERSON': 'person',
            'FAC': 'landmark',
            'ORG': 'organization',
            'GPE': 'place',
            'LOC': 'location',
            'EVENT': 'event',
            'WORK_OF_ART': 'artwork',
            'LAW': 'law',
            'DATE': 'date',
            'TIME': 'time',
            'PERCENT': 'percent',
            'MONEY': 'money',
            'QUANTITY': 'quantity',
            'CARDINAL': 'number'
}

ent_order = {'PERSON': 8,
            'FAC': 2,
            'ORG': 1,
            'GPE': 6,
            'LOC': 7,
            'EVENT': 3,
            'WORK_OF_ART': 5,
            'LAW': 4,
            'DATE': 9,
            'TIME': 10,
            'PERCENT': 12,
            'MONEY': 11,
            'QUANTITY': 13,
            'CARDINAL': 14,
}

drop_ents = ['NORP', 'PRODUCT', 'LANGUAGE','ORDINAL']

preprocess = Cleaner(replacements)

def process(in_doc):
    count = 0
    out_doc = ""
    doc = nlp(in_doc)
    if doc._.language['language'] != 'en':
        return None
    colon_count = 0
    for sent in doc.sents:
        text = sent.text
        if ':' in text:
            colon_count += 1
        if not re.search('[.?!] *$', text) or re.search(r'(?i)you', text): # direct appeal to reader or not a sentence
            continue
        out_doc += sent.text + ' '
        count += 1
    if count < 13 or colon_count > 6: # too short for training or likely contains many unquoted quotations
        return None
    ents = list(set([ent for ent in doc.ents if ent.label_ not in drop_ents]))
    ents = sorted(ents, key=lambda ent: ent_order[ent.label_])
    for ent in ents:
        if ent.text[0] == '$':
            pattern = r'{}*\b'.format(ent.text) # match money strings, not first word
        else:
            pattern = r'\b{}\b'.format(ent.text) # only match pattern as a word, not part of a word
        out_doc = re.sub(pattern, entities.get(ent.label_, ent.text), out_doc)
    ents2 = set([ent for ent in nlp(out_doc).ents if ent.label_ == 'PERSON'])
    for ent in ents2:
        pattern = r'\b{}\b'.format(ent.text) 
        out_doc = re.sub(pattern, 'person', out_doc)
    return out_doc

def convert_quotes(qq):
    num = 0
    if qq[-2] in ['.', '?', '!']:
        punct = qq[-2]
    else:
        punct = ''
    length = len(qq.split())
    if length <= 2:
        num = 1
    elif length <= 12:
        num = 2
    elif length <= 25:
        num = 3
    else:
        num = 4
    return 'quote ' * num + punct

def reformat(article):
    if not article:
        return None
    if type(article) is not str:
        return None
    text = unidecode(article)
    if text.count('\N{QUOTATION MARK}') % 2 != 0:
        return None
    text = preprocess.clean(text)
    text = re.sub(r'^(.{0,50})\(\w+\)', ' ', text) # delete dateline
    text = re.sub(r'\|.*\|', ' ', text) # delete category headers, bylines, etc. between pipe symbols
    text = re.sub(r'\S*@\S+', 'email', text) # replace email address or Twitter handle with "email"
    text = re.sub(r'[-a-zA-Z0-9@:%_\+.~#?&\/=]{2,256}\.[a-z]{2,4}(\/[-a-zA-Z0-9@:%_\+.~#?&\/=]*)?', ' website',
                  text) # URLs
    text = re.sub('[\[\(][^\[\(]*[\]\)]', '', text) # delete text inside parentheses or brackets
    text = re.sub(r"\b(\w*)n't", lambda m: m.group(1) + ' not', text) # replace remaining "xxn't" contractions with "xx not"
    text = re.sub(r'("[^"]*")', lambda m: convert_quotes(m.group(1)), text) # replace quoted text
    text = re.sub(r"^'|'$|(?<= )'|(?<!s)'(?= )", '"', text) # replace single quotes, but not apostrophes, with double quotes
    if text.count('\N{QUOTATION MARK}') % 2 != 0: # unbalanced quotation marks would cause improper processing 
        return None
    text = re.sub(r'("[^"]*")', lambda m: convert_quotes(m.group(1)), text) # replace quoted text
    text = re.sub(r'(?i)please share this.*', '', text)
    text = re.sub(' +', ' ', text) # reduce all multiple spaces to single spaces
    try:
        output = process(text)
    except:
        output = None
    return output

def recombine(array):
    return [' '.join(' '.join(sent) for sent in array)][0]

def create_data_matrix(data, max_sentences=max_sentences, max_words=max_words, max_vocab=max_vocab,
                      word_index=None):
    data_matrix = np.zeros((len(data), max_sentences, max_words), dtype='int32')
    for i, article in enumerate(data):
        for j, sentence in enumerate(article):
            if j == max_sentences:
                break
            k = 0
            for word in sentence:
                if k == max_words:
                    break
                ix = word_index.get(word.lower())
                if ix is not None and ix < max_vocab:
                    data_matrix[i, j, k] = ix
                k = k + 1
    return data_matrix
