# Named Entity Recognition with BERT and spaCy

NER system for extracting structured entities from unstructured text — built and compared across three different approaches as part of an NLP course at VIT Chennai.

## What It Extracts

17 entity types using BIO tagging: person names (B-PER), locations (B-GEO), geopolitical entities (B-GPE), organisations (B-ORG), time expressions (B-TIM), artifacts (B-ART), natural phenomena (B-NAT), events (B-EVE), plus I-* continuation tags for each.

## Dataset

**NER Labelled Corpus**
- 1,048,575 words total, BIO-tagged
- 80/20 train/test split: 838,860 training tokens, ~209,715 test tokens
- Entity examples: `London` → B-GEO, `Iraq` → B-GEO, `British` → B-GPE
- `ner_dataset.csv` — the training data is in the repository

## Models Compared

### 1. BERT Fine-tuned (`bert-base-cased`)
Fine-tuned `BertForTokenClassification` on the full dataset.

| Metric | Score |
|---|---|
| F1-Score | **0.7937** |
| Precision | 0.8276 |
| Recall | 0.7624 |
| Eval Loss | 0.1713 |

Training config: 1 epoch · learning rate 1e-4 · batch size 32 · 17 output classes

### 2. Baseline Sequence Labelling
Traditional sequence tagger without contextual embeddings. Lower accuracy on ambiguous and multi-word entities.

### 3. Custom spaCy NER Pipeline (`Spacy_Custom_NER_Youtube.ipynb`)
Trained on domain-specific examples with custom entity definitions. Useful for adding entity types not covered by the base BERT model.

**Key observation:** BERT's contextual embeddings handle entities that rule-based and classic sequence models consistently miss — for example, `Bangalore` used as a location vs. a company name depending on sentence context.

## Repository Contents

| File | Description |
|---|---|
| `BERT_NAMED_ENTITY_RECOGNITION.ipynb` | BERT fine-tuning, evaluation, entity extraction |
| `NAMED ENTITY RECOGNITION.ipynb` | Baseline sequence labelling approach |
| `Spacy_Custom_NER_Youtube.ipynb` | Custom spaCy NER training pipeline |
| `ner_dataset.csv` | Labelled training dataset (1M+ tokens) |
| `ResearchPaper_bert.pdf` | BERT NER research paper |
| `21BCE1950_5828_NLP_DA1SURVEY.pdf` | NLP survey report (DA1) |
| `21BCE1950_5828_NLP_DA_1 (1).pdf` | DA1 alternate submission |
| `21BCE1950_Omprakash_NLP_DA3.pdf` | Final NLP report (DA3) |
| `NLP_DA-3-1.pdf` | DA3 reviewed submission |

## Stack

Python · HuggingFace Transformers · BERT (`bert-base-cased`) · spaCy · scikit-learn · pandas · Jupyter Notebook
