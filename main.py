import spacy
import gensim
import nltk
import wordfreq
import re
import numpy as np
import spacy
from spacy.lang.pt.examples import sentences

gensim.models.Word2Vec

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("pt_core_news_sm")

import nltk

STOPWORDS = nltk.corpus.stopwords.words('portuguese')
DROP_FIRST = ['to']
DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
DIGIT_RE = re.compile(r'[0-9]')

lp = spacy.load("pt_core_news_sm")
doc = nlp(sentences[0])



def standardized_uri(language, term):
    """
    Get a URI that is suitable to label a row of a vector space, by making sure
    that both ConceptNet's and word2vec's normalizations are applied to it.

    'language' should be a BCP 47 language code, such as 'en' for English.

    If the term already looks like a ConceptNet URI, it will only have its
    sequences of digits replaced by #. Otherwise, it will be turned into a
    ConceptNet URI in the given language, and then have its sequences of digits
    replaced.
    """
    if not (term.startswith('/') and term.count('/') >= 2):
        term = _standardized_concept_uri(language, term)
    return replace_numbers(term)

def filter(tokens):
    """
    Given a list of tokens, remove a small list of English stopwords. This
    helps to work with previous versions of ConceptNet, which often provided
    phrases such as 'an apple' and assumed they would be standardized to
	'apple'.
    """
    non_stopwords = [token for token in tokens if token not in STOPWORDS]
    while non_stopwords and non_stopwords[0] in DROP_FIRST:
        non_stopwords = non_stopwords[1:]
    if non_stopwords:
        return non_stopwords
    else:
        return tokens

def replace_numbers(s):
    """
    Replace digits with # in any term where a sequence of two digits appears.

    This operation is applied to text that passes through word2vec, so we
    should match it.
    """
    if DOUBLE_DIGIT_RE.search(s):
        return DIGIT_RE.sub('#', s)
    else:
        return s

def _standardized_concept_uri(language, term):
    if language == 'en':
        token_filter = filter
    else:
        token_filter = None
    language = language.lower()
    norm_text = _standardized_text(term, token_filter)
    return '/c/{}/{}'.format(language, norm_text)

def _standardized_text(text, token_filter):
    tokens = simple_tokenize(text.replace('_', ' '))
    if token_filter is not None:
        tokens = token_filter(tokens)
    return '_'.join(tokens)

def simple_tokenize(text):
    """
    Tokenize text using the default wordfreq rules.
    """
    return wordfreq.tokenize(text, 'xx')


# sentence = sentences[0]
sentence = 'Fui ir irei viajar aluna de vocês no período de 2010-2014 no curso de tecnologia em logística. E em 2014 -2016 em gestão empresarial. No campus Av. Souza Naves em Curitiba.'
# sentence = standardized_uri('pt', sentence)
tokens = sentence.split(' ')
tokens = filter(tokens)



# temp1 = gensim.matutils.unitvec(standardized_uri('pt', 'fui'))
# temp2 = gensim.matutils.unitvec(standardized_uri('pt', 'ir'))

doc = nlp(sentence)

new_sentence = ' '.join(tokens)
#
# # Process whole documents
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")
doc = nlp(new_sentence)
#
# # Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

for entity in doc.ents:
    print(entity.text, entity.label_)

numberbatch = gensim.models.KeyedVectors.load(
    'conceptnet.model',
    mmap='r'
)

temp1 = numberbatch.numeric_representation(standardized_uri('pt', 'boi'))
temp2 = numberbatch.numeric_representation(standardized_uri('pt', 'frango'))

resut = 1 - np.dot(temp1, temp2)

# gensim.models.KeyedVectors(standardized_uri('pt', 'ir'))

similarity = numberbatch.similarity(standardized_uri('pt', 'boi'), standardized_uri('pt', 'frango'))

# numberbatch.save('conceptnet.model')

keys = numberbatch.key_to_index

for token in tokens:
    try:
        print('='*100)

        similarity = numberbatch.similarity(standardized_uri('pt', 'ir'), standardized_uri('pt', 'fui'))

        most_similar = numberbatch.most_similar(positive=[standardized_uri('pt', token)], negative=[], topn=100)



        temp_vector = []

        for temp_token in tokens:
            try:
                standardized_temp_token = standardized_uri('pt', temp_token)
                if (standardized_temp_token in keys):
                    temp_vector.append(standardized_temp_token)
            except Exception as ex2:
                pass

        distances = numberbatch.distances(standardized_uri('pt', token),
                                          temp_vector)


        print('distances', distances)
        # print('Similaridade:{}'.format(similarity))
        print('{} - Mais similares:{}'.format(token, most_similar))

        print('='*100)
    except Exception as ex:
        pass
#
# # Find named entities, phrases and concepts




