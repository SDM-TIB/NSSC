import pprint
import time
from ner import extract_entities
import logging
import traceback
from bk_indexer import Indexer
from bk_definitions import SemanticInfo
from llm import LargeLanguageModel
from models import  Candidate
from optimization import filter_by_similarity
from tqdm import tqdm


MODEL = "gpt-35-turbo"
MAX_TOKENS = 100  # Should be adjust depende on the query
TEMPERATURE = 0.5  # Low temperature to be the more precciss as posible
PARAMS = {'temperature':TEMPERATURE, 'max_tokens':MAX_TOKENS}
LLM = LargeLanguageModel(MODEL, 3, **PARAMS ) 

INDEXER = Indexer(id_query="filtered_boosted")
SEMANTIC = SemanticInfo()


def search_term(term, searcher: Indexer, optimize=True) -> list[Candidate]:
    results = searcher.search(term)
    
    candidates = [Candidate(r['cui'], r['sgroup'], r['label']) for r in results]

    if len(candidates) > 0:

        langs = set([ r['lang'] for r in results])

        if 'SPA' not in langs and 'ENG' not in langs:
            print("The term is founded in oder language", term)
            print(langs)
            pprint.pprint(list(map(str, candidates)))
        
        if optimize:
            return filter_by_similarity(term, candidates)
        else:
            return candidates
    else:
        print("BK dont found: ", term)
        return []

def get_umls_definition(umls_definitions, candidates):

    for c in candidates:
        if c.cui not in umls_definitions:
            definitions = SEMANTIC.get_cui_definition(c.cui)
            first_def = definitions[0][0] if definitions else None

            umls_definitions[c.cui] = first_def

        c.set_definition(umls_definitions[c.cui])


def disambiguate(term:str, context:str, candidates):

    result = LLM.ask_llm(term, context, candidates)
    answer = result["answer"]

    revised_candidates = answer.get('candidates', False)

    term_transleted = answer.get('translation', False)

    return revised_candidates, term_transleted, result


def select_best_candidates(term: str, context:str, candidates: list[Candidate], umls_definitions: dict) -> list[str]:

    api_usage = []

    print('\n Term:', term, '\n')

    if len(candidates) == 0:
        print("No candidates")
        return {"candidates":False, "usage": False}
    
    if len(candidates) == 1:  # Best candidates is alredy founded
        return {"candidates": [candidates[0].cui], "usage": False}
    
    # Desambiguate with LLM
    try:

        get_umls_definition(umls_definitions, candidates)

        revised_candidates, term_transleted, result = disambiguate(
            term, context, candidates)
        
        api_usage.append(result)

        if revised_candidates == False:

            print(term_transleted)

            new_candidates = search_term(term_transleted, INDEXER)

            if len(new_candidates) == 0:
                return {"candidates":False, "usage": False}
             
            if len(candidates) == 1:  # Best candidates is alredy founded
                return {"candidates": [candidates[0].cui], "usage": False}
            
            # Try in English
            get_umls_definition(umls_definitions, new_candidates)

            revised_candidates, _, result = disambiguate(
                term_transleted, context, new_candidates)
            
            api_usage.append(result)

        print(revised_candidates)

        return {"candidates": revised_candidates, "usage": api_usage}

    except Exception as e:
        logging.error(str(e))
        logging.error(traceback.format_exc())
        time.sleep(10)
        return {"candidates":False, "usage": False, "why": str(e), "term": term}

 
def link2umls(terms_context):
    '''

    '''

    # Background Knowledge
    term_candidates_context = [(term, context, search_term(term, INDEXER)) for term, context in tqdm(terms_context, "Quering BK")]

    # Algorith that select the best
    SEMANTIC.open_connection()
    umls_definitions = {}

    terms_bests = {term: select_best_candidates(
        term, context, candidates, umls_definitions) for term, context, candidates in term_candidates_context}
    
    SEMANTIC.close_connection()

    return terms_bests


def main():
    '''
    Map a free-text of cancer diagnosis and tratments concepts to UMLS

    In: a string that represent the clinical note
    Out: The best cuis that represent the clinical note

    Pipeline: 
    raw text -> (Preprocesing) -> clean text -> (NER) -> set of tuples (entity, label) -> terms2normalize -> 
    (elastic search) -> list of candidates for each term -> remove duplicates for each term -> 
    (mySQL - Definitions) -> prepare best candidates ->
    (LLM-OPENAI) -> best cui for each term -> JSON structured and normalized

    '''
    text = "carcinoma de mama izquierda microinfiltrante tmicn0m0 + her2 negativo. premenopausica tratada con mastectomia."
    
    ents = extract_entities(text)

    best_terms = link2umls(ents)

    pprint.pprint(best_terms)

if __name__ == '__main__':
    main()
