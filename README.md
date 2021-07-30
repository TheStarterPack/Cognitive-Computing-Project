# Action Embeddings (CC-Project SS 21) 


## Installation

Supported python version: 3.9

We recommend using a virtual environment especially to avoid dependency problems. 

Create virutal environment:

```
python -m pip install --user virtualenv

python -m venv env

source env/bin/activate (For Windows: env\Scripts\activate)
```

Install requirements:

```
pip install -r requirements.txt
```

## Motivation

We realised a pipeline for the creation and evaluation of action embeddings. Additionally we assess if arithmetic relations can also be observed likewise in action embeddings. After that we want to evaluate the created embeddings by considering metrics and application in furthermore tasks. Finally we talk about limitations and problems that occur during the development process.



## What are Embeddings? (Till)


Action embeddings are similar to word embeddings. The central motivation behind word embeddings is to represent the contextual relations of words in a multi dimensional vector space. Due to their contextual information embeddings are often used in further specific behavior understanding/generation tasks. The contextual meaning is realised by the arithmetic of the vector space. For example we have the relation capital:

*Capital = France - Paris*

We could now use the relation to gain information.

*Germany + Capital = Berlin*

For the classification of a word type, the surrounding words could be used directly as an input for a neuronal Network. Alternatively the words are replaced by the corresponding embeddings and the results get better in many cases. 

In a nutshell the embedding vectors are trained by taking sequences from the corpus. For each sequence embeddings of the word in the middle and the surrounding words get customized to be more similar to each other. As a result the whole training process yield embedding vectors which are more similar to each other if the corresponding words occure in the same context. Likewise are action embeddings are created.

## Our corpus/dataset (Lukas)

## Training Process (Augustin)

## Evaluation 

### Clustering

### Embedding Arithmetic

### Prediction





Start the Bullseye prediction

```
python prediction.py
```

If you want to change the hyperparameters you can edit the variables at module level.

## Installation/Usage Guide

### Installation

### Training

### Clustering

### Arithmetic

### Prediction
