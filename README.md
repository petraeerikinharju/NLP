# NLP

### Instructions

Download everything in this repository to your computer. Note that if the code doesn't run, make sure that all the packages are compatible. The package versions used while making this code are marked in *Versions.txt*.

*Semantic_sim.py* contains almost all of the methods used in this code and *main.py* contains the calculations for the majority of the assignment. Running the code takes a while. On this computer it took around 1,5 minutes. Sem_Sim_Notebook.ipynb contains a notebook of the code, so you can also see the results straight from there. 

If you want to use larger sets of the SemRel2024 datasets, modify the code on line 100. (In the for-loop the line 
`sentences, labels = extract_sentences_and_labels(dataset)`
add another variable to the method. See the method definition above this for-loop.)

The *Interface.py* contains a simple interface to check similarities between two input sentences.

The code in folder SOC_PMI is from https://github.com/pritishyuvraj/SOC-PMI-Short-Text-Similarity-
