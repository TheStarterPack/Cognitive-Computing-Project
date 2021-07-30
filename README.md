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


Likewise word embeddings we want that action embeddings can be used in high level machine-learning tasks. Like in the introduction mentioned word embeddings are used as an input for next word or word type classification. As an equivalent we made-up the task of unparameterized action prediction. This task is defined by predicting the unparameterized action given the embeddings of the surrounding actions of an sequence. For example we have this action sequence:

[WALK] <food_food>, [GRAB] <food_food>, [FIND] <freezer>, [PUTIN] <food_food> <freezer> , [CLOSE] <freezer>

The model would receive the embeddings of the actions [WALK] <food_food> (1), [GRAB] <food_food> (1), [PUTIN] <food_food> (1) <freezer> (1), [CLOSE] <freezer> (1) and should predict the unparameterized action [FIND]. 

Start prediction

```
python prediction.py
```

If you want to change the hyperparameters you can edit the variables at module level.


We used the whole corpus to create input embedding sequences and unparameterized action as prediction target. The training and evaluation was maded with a randomized test/train split of 0.80/0.20%, batch size of 64, 10 epochs and a learning rate of 0.005.

The architecture consists of three layers. An LSTM, Batch Normalization and Linear Layer. A LSTM Layer is a recurrent layer which receives an timestep sequence (here embedding sequence) and learns during the train process the relations between the time step by including a hidden state into the processing of each timestamp. As a result, it's possible to learn the relation of the timesteps among themselves. The hidden state of the last timestamp is normalized by the batch normalization layer to avoid overfitting to high frequent classes. After that the final dense layer maps the normalized hidden state to the number of classes.

The results are not that good. We achieved an accuracy of 60% in the train and 40% in the test set. A reason for that could be that the network architecture is not suitable or that the train process of the embeddings is more focused on the parameters instead of the unparameterized action. Another possible reason could be that the corpus is to small to learn embeddings or high level tasks based on that.

### Training

### Clustering

### Arithmetic

### Prediction
