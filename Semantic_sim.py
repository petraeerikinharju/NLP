import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import csv
import pandas as pd
import nltk
import spacy
import numpy as np
import tensorflow_hub as hub
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from datasets import load_dataset
from SOC_PMI.main import similarity

nlp = spacy.load("en_core_web_md")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
stopwords = list(set(nltk.corpus.stopwords.words('english')))

### TASK 1 ###

def preProcess(sentence):
    """Preprocess a single sentence by tokenizing, converting to lowercase, and removing stop words."""
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    filtered_sentence = [word for word in tokenized_sentence if word not in stopwords]
    return filtered_sentence


def sim1(sentence_list):
    """Calculate sentence-to-sentence similarity using TF-IDF and WordNet similarity."""
    computed_similarities = []
    tf = TfidfVectorizer(use_idf=True)

    for T1, T2 in sentence_list:
        words1 = preProcess(T1)
        words2 = preProcess(T2)

        tf_matrix = tf.fit_transform([' '.join(words1), ' '.join(words2)])
        
        sim_score = cosine_similarity(tf_matrix[0:1], tf_matrix[1:2])[0][0]
        computed_similarities.append(round(sim_score, 2))

    return computed_similarities

def read_from_csv(file_path):
    """Read sentences and the corresponding similarity scores from a csv file using pandas."""
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    
    # Ensure the DataFrame has the correct number of columns
    if df.shape[1] != 3:
        raise ValueError("CSV file must contain exactly 3 columns.")

    # Extract sentences and scores
    sentences = list(zip(df.iloc[:, 0].str.strip(), df.iloc[:, 1].str.strip()))
    scores = df.iloc[:, 2].astype(float).tolist()
    
    return sentences, scores

### TASK 2 ###

def antonym(token):
    """Return the antonym of a given token using WordNet."""
    synsets = wn.synsets(token)
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.antonyms():
                return lemma.antonyms()[0].name()  # Return the first antonym found
    return token  # Return the original token if no antonym is found

def preprocess_with_negation_and_entities(sentence):
    """Preprocess a sentence to handle negation and extract noun entities."""
    doc = nlp(sentence)
    tokens = [token for token in doc]  # Keep token objects
    
    # Check for named entities
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    if len(named_entities) == 0:  # No named entities
        negated_tokens = []
        for i, token in enumerate(tokens):
            if token.text.lower() in ['not', 'no', 'never']:  # Negation found
                if i + 1 < len(tokens) and tokens[i + 1].pos_ in ['ADJ', 'ADV']:
                    negated_tokens.append(antonym(tokens[i + 1].text))  # Convert to antonym
                else:
                    negated_tokens.append(token.text)  # Keep the negation
            else:
                negated_tokens.append(token.text)

        # Filter to nouns and convert to nouns using WordNet
        noun_tokens = []
        for token in negated_tokens:
            pos = wn.synsets(token)
            if pos:  # If token exists in WordNet
                noun_tokens.append(pos[0].lemmas()[0].name())  # Convert to noun

        return [nt for nt in noun_tokens if nt not in stopwords], named_entities

    else:  # Handle named entities
        # Discard any named entity not present in both sentences
        tokens = [token.text for token in tokens if token.text not in [ent[0] for ent in named_entities]]
        return [token for token in tokens if token not in stopwords], named_entities

def wu_palmer_similarity(noun_tokens1, noun_tokens2):
    """Calculate the average Wu-Palmer similarity between two lists of nouns."""
    similarities = []
    for noun1 in noun_tokens1:
        for noun2 in noun_tokens2:
            syn1 = wn.synsets(noun1)
            syn2 = wn.synsets(noun2)
            if syn1 and syn2:
                similarity = wn.wup_similarity(syn1[0], syn2[0])
                if similarity is not None:
                    similarities.append(similarity)
    return np.mean(similarities) if similarities else 0.0

def sim2(sentence_list):
    """Calculate sentence-to-sentence similarity as described."""
    computed_similarities = []
    
    for T1, T2 in sentence_list:
        nouns1, named_entities1 = preprocess_with_negation_and_entities(T1)
        nouns2, named_entities2 = preprocess_with_negation_and_entities(T2)
        
        # Compute Wu-Palmer similarity for nouns
        if not named_entities1 and not named_entities2:  # No named entities in both
            sim_score = wu_palmer_similarity(nouns1, nouns2)
        elif named_entities1 and named_entities2:  # Named entities present in both
            named_entity_sim = cosine_similarity(
                [nlp(ent[0]).vector for ent in named_entities1],
                [nlp(ent[0]).vector for ent in named_entities2]
            ).max()  # Get max cosine similarity
            sim_score = 0.5 * named_entity_sim + 0.5 * wu_palmer_similarity(nouns1, nouns2)
        else:  # Only one sentence has named entities
            sim_score = wu_palmer_similarity(nouns1, nouns2)

        computed_similarities.append(round(sim_score, 2))
        
    return computed_similarities

### TASK 4 ###
def compute_similarity_doc2vec(sentence_list, epochs=200):
    """Train a Doc2Vec model using a list of sentence pairs."""

    tagged_data = [TaggedDocument(words=preProcess(s[0]) + preProcess(s[1]), tags=[str(i)]) for i, s in enumerate(sentence_list)]

    doc2vec_model = Doc2Vec(vector_size=200, alpha=0.025, min_alpha=0.00025, min_count=1, dm=1, epochs=epochs)
    doc2vec_model.build_vocab(tagged_data)
    doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    computed_similarities_doc2vec = []
    for sentence1, sentence2 in sentence_list:
        try:
            vec1 = doc2vec_model.infer_vector(preProcess(sentence1))
            vec2 = doc2vec_model.infer_vector(preProcess(sentence2))
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            computed_similarities_doc2vec.append(similarity)
        except Exception as e:
            print(f"Error processing pair ({sentence1}, {sentence2}): {e}")

    return computed_similarities_doc2vec, doc2vec_model

def compute_spacy_embeddings(sentence_list):
    """Compute SpaCy embeddings for a list of sentence pairs."""
    computed_similarities = []
    for sentence1, sentence2 in sentence_list:
        # Generate embeddings using SpaCy
        vec1 = nlp(sentence1).vector
        vec2 = nlp(sentence2).vector
        # Calculate cosine similarity
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        computed_similarities.append(similarity)
    return computed_similarities

def compute_distilbert_embeddings(sentence_list):
    """Compute DistilBERT embeddings for a list of sentence pairs."""
    computed_similarities = []
    for sentence1, sentence2 in sentence_list:
        inputs1 = tokenizer(sentence1, padding=True, truncation=True, return_tensors='pt')
        inputs2 = tokenizer(sentence2, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs1 = distilbert_model(**inputs1)
            outputs2 = distilbert_model(**inputs2)

        vec1 = outputs1.last_hidden_state.mean(dim=1).squeeze().numpy()
        vec2 = outputs2.last_hidden_state.mean(dim=1).squeeze().numpy()

        similarity = cosine_similarity([vec1], [vec2])[0][0]
        computed_similarities.append(similarity)
    return computed_similarities

def compute_similarity_use(sentence_list):
    """Compute cosine similarity for a list of sentence pairs using the Universal Sentence Encoder."""
    computed_similarities = []
    
    for sentence1, sentence2 in sentence_list:
        embeddings = embed([sentence1, sentence2]).numpy()
        
        similarity = cosine_similarity(embeddings)[0, 1]
        computed_similarities.append(similarity)
        
    return computed_similarities
