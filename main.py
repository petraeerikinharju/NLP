from Semantic_sim import *
import os
import pandas as pd

os.chdir(os.path.dirname(__file__))
sentences_stss, human_similarities = read_from_csv("STSS-131.csv")

#sim1
computed_similarities_1 = sim1(sentences_stss)
print(f"List lengths: {len(sentences_stss)}, {len(human_similarities)}, {len(computed_similarities_1)}")
pearson_coeff_1, _ = pearsonr(human_similarities, computed_similarities_1)
print(f"Pearson correlation coefficient Sim1: {pearson_coeff_1:.2f}")

# Test sim2 with 10 sentence pairs
test_pairs = [
    ("The city was noisy.", "The forest was silent."),
    ("Did you finish your homework?", "Have you completed your assignments?"),
    ("The cat sat on the warm windowsill.", "A cat rested on a cozy window ledge."),
    ("He does not like apples.", "He dislike apples."),
    ("The food was delicious.", "The meal was tasty."),
    ("The quick brown fox jumps over the lazy dog.", "A quick fox leaps over a lazy hound."),
    ("She is not happy with the results.", "She is sad with the results."),
    ("Apple Inc. released a new product.", "Google LLC announced their latest software."),
    ("He did not find the answer quickly.", "He found the answer slowly."),
    ("NASA announced a new space mission.", "The European Space Agency confirmed another mission."),
]
computed_similarities = sim2(test_pairs)
for (S1, S2), sim2_score in zip(test_pairs, computed_similarities):
    print(f"Similarity between:\n'{S1}'\nand\n'{S2}'\nis: {sim2_score:.4f}\n")

computed_similarities_2 = sim2(sentences_stss)

'''
df = pd.DataFrame({
    'Sentence 1': [s[0] for s in sentences_stss],
    'Sentence 2': [s[1] for s in sentences_stss],
    'Human Similarity': human_similarities,
    'Computed Similarity Sim1': computed_similarities_1,
    'Computed Similarity Sim2': computed_similarities_2
})

 You can see the table in the GitHub
df.to_excel('similarities.xlsx', index=False)
'''

pearson_coeff_2, _ = pearsonr(human_similarities, computed_similarities_2)

print(f"Pearson correlation coefficient Sim2: {pearson_coeff_2:.2f}")

# State-of-the-art embeddings:

computed_similarities_doc2vec, model = compute_similarity_doc2vec(sentences_stss)
pearson_coeff_doc2vec = pearsonr(human_similarities, computed_similarities_doc2vec)[0]
print(f"Pearson correlation coefficient with Doc2Vec: {pearson_coeff_doc2vec:.2f}")

computed_similarities_spacy_e = compute_spacy_embeddings(sentences_stss)
pearson_coeff_spacy_e = pearsonr(human_similarities, computed_similarities_spacy_e)[0]
print(f"Pearson correlation coefficient with SpaCy embedding: {pearson_coeff_spacy_e:.2f}")

computed_similarities_distilbert_e = compute_distilbert_embeddings(sentences_stss)
pearson_coeff_distilbert_e = pearsonr(human_similarities, computed_similarities_distilbert_e)[0]
print(f"Pearson correlation coefficient with DistilBERT embedding: {pearson_coeff_distilbert_e:.2f}")

computed_similarities_use = compute_similarity_use(sentences_stss)
pearson_coeff_use, _ = pearsonr(human_similarities, computed_similarities_use)
print(f"Pearson correlation coefficient with Universal Sentence Encoder: {pearson_coeff_use:.2f}")

# Another dataset:

ds = load_dataset("SemRel/SemRel2024", "eng")
datasets = ["train", "test", "dev"]

def extract_sentences_and_labels(dataset_name, num_samples=200):
    '''Extract sentences and labels from a dataset from SemRel2024'''
    '''You can change the number of samples. It is by default set to 200 to ease the computation'''
    dataset = ds[dataset_name].shuffle(seed=42)
    max_samples = min(num_samples, len(dataset))
    dataset = dataset.select(range(max_samples))
    
    sentences = []
    labels = []

    for item in dataset:
        sentence1 = item['sentence1'].strip()
        sentence2 = item['sentence2'].strip()
        label = float(item['label'])

        sentences.append((sentence1, sentence2))
        labels.append(label)
        
    print(f"Extracted {len(sentences)} sentence pairs from the {dataset_name} set.")
    return sentences, labels 

results = {}
stored_scores = {}

for dataset in datasets:
    '''Use all the previous methods for SemRel2024 datasets'''

    sentences, labels = extract_sentences_and_labels(dataset)

    sim1_scores = sim1(sentences)
    sim2_scores = sim2(sentences)
    doc2vec_scores, model = compute_similarity_doc2vec(sentences, 150)
    spacy_scores = compute_spacy_embeddings(sentences)
    distilbert_scores = compute_distilbert_embeddings(sentences)
    use_scores = compute_similarity_use(sentences)

    stored_scores[dataset] = {
        'doc2vec': doc2vec_scores,
        'spacy': spacy_scores,
        'distilbert': distilbert_scores
    }

    # Calculate Pearson correlation coefficients
    sim1_corr = pearsonr(sim1_scores, labels)[0]
    sim2_corr = pearsonr(sim2_scores, labels)[0]
    doc2vec_corr = pearsonr(doc2vec_scores, labels)[0]
    spacy_corr = pearsonr(spacy_scores, labels)[0]
    distilbert_corr = pearsonr(distilbert_scores, labels)[0]
    use_corr = pearsonr(use_scores, labels)[0]
        
    results[dataset] = {
        'sim1': sim1_corr,
        'sim2': sim2_corr,
        'doc2vec': doc2vec_corr,
        'SpaCy': spacy_corr,
        'DistilBERT': distilbert_corr,
        'use': use_corr
    }

for dataset, correlations in results.items():
    print(f"{dataset.capitalize()} Results:")
    for method, corr in correlations.items():
        print(f"  {method}: {corr:.2f}")
        
### Weighted state-of-the-art
weights = [0.25, 0.25, 0.5]

for dataset in datasets:
    '''Use all the previous methods for SemRel2024 datasets'''
    sentences, labels = extract_sentences_and_labels(dataset)
    
    doc2vec_scores = stored_scores[dataset]['doc2vec']
    spacy_scores = stored_scores[dataset]['spacy']
    distilbert_scores = stored_scores[dataset]['distilbert']
    
    ensemble_scores = [
        sum(w * sim for w, sim in zip(weights, similarities))
        for similarities in zip(doc2vec_scores, spacy_scores, distilbert_scores)
    ]
    ensemble_corr = pearsonr(ensemble_scores, labels)[0]
    results[dataset] = {
        'Ensemble': ensemble_corr
    }
    
for dataset, correlations in results.items():
    print(f"{dataset.capitalize()} Results:")
    for method, corr in correlations.items():
        print(f"  {method}: {corr:.2f}")
        
# Similarity from GitHub SOC-PMI-Short-Text-Similarity

SOC_similarities = []
for S1, S2 in sentences_stss:
    sim_score = similarity(S1, S2) # Call the similarity from the provided repository
    SOC_similarities.append(sim_score)
    
SOC_coefficient = pearsonr(SOC_similarities, human_similarities)[0]
print(f"Pearson correlation coefficient SOC-PMI-Short-Text-Similarity-: {pearson_coeff_1:.2f}")
