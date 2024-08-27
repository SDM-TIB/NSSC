import difflib as dif
import spacy

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

#model = SentenceTransformer('all-MiniLM-L6-v2')
#nlp = spacy.load("es_core_news_lg", disable=["tagger", "parser", "ner", 'morphologizer', 'attribute_ruler', 'lemmatizer'])


def similarity_string(term_search: str, term_candidate: str) -> float:
    return dif.SequenceMatcher(None, term_search, term_candidate).ratio()

def similarity_semantic(term_search: str, term_candidate: str) -> float:

    doc1 = nlp(term_search)
    doc2 = nlp(term_candidate)

    return doc1.similarity(doc2)

esq
def similarity_transformer(term_search: str, term_candidate: str):

    sentences = [term_search, term_candidate]

    embedded_list = model.encode(sentences)

    return float(cos_sim(embedded_list[0], embedded_list[1])[0][0])



