import spacy
from preprocesing import Preprocesing

nlp = spacy.load("es_breast_cancer_ehr_ner")
pre = Preprocesing("./static/acronims.json")

entity_mapping = {
    'Cancer Concept': 'CANCER_CONCEPT',
    'Cancer Type': 'CANCER_TYPE',
    'Cancer Expansion': 'CANCER_EXP',
    'Cancer Location': 'CANCER_LOC',
    'Cancer Metastasis': 'CANCER_MET',
    'Cancer Recurrence': 'CANCER_REC',
    'Cancer Subtype' : 'CANCER_SUBTYPE',
    'Molecular Marker': 'MOLEC_MARKER',
    'Cancer Stage': 'CANCER_STAGE',
    'TNM': 'TNM',
    'Treatment Name': 'TRAT',
    'Treatment Schema': 'TRAT_SHEMA',
    'Treatment Drug': 'TRAT_DRUG',
    'Treatment Frequency': 'TRAT_FREQ',
    'Treatment Quantity': 'TRAT_QUANTITY',
    'Surgery': 'SURGERY',
}

inverse_mapping = {v: k for k, v in entity_mapping.items()}


NORMALIZE_ENTS = {"CANCER_CONCEPT", 'CANCER_EXP', 'CANCER_LOC', 'CANCER_TYPE',
                  'CANCER_SUBTYPE', 'CANCER_INTRATYPE', 'SURGERY', 'TRAT', 'TRAT_SHEMA', 'TRAT_DRUG'}

def extract_entities(text: str):
    text_pre = pre.fix(text)
    doc = nlp(text_pre)
    ents = [(ent.text, inverse_mapping[ent.label_]) for ent in doc.ents if ent.label_ in NORMALIZE_ENTS]

    return ents


if __name__ == '__main__':
    ents = extract_entities("Paciente con carcinoma de mama ductal 28/02/2023")
    print(ents)