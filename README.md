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

For training the action embeddings the Virtual Home’s ‘Activity Knowledge Base’ (AKB) (http://virtual-home.org/tools/explore.html) was used. 
The AKB consists of many action sequences which were collected for training robots. 

Each action sequence consists of a series of ordered actions and aims to achieve some sort of goal. 
Additionally, for each sequence there are a title describing the goal, a natural language sentence describing the approach for achieving this goal and a set of conditions which the environment should fulfill in order for the action sequence to be executable. 

In our work we discarded all info except for the action sequences and actions themselves. Each action has a title describing the action, for example “Walk”, “Grab” or “Find”. Optionally actions may also have a list of parameters, or targets, on which the action is performed, for example: “Walk (Table)” for walking to the table or “PutOn (Table, Plate)” for putting a plate on the table. 

Example of an action sequence:

```
Put groceries in Fridge
I put my groceries into the fridge.

[WALK] <food_food> (1)
[GRAB] <food_food> (1)
[FIND] <freezer> (1)
[PUTIN] <food_food> (1) <freezer> (1)
[CLOSE] <freezer> (1)
```

When training the action embeddings we considered two actions to be the same if their titles and parameters are exactly the same.
This means that the order of parameters, which are represented as a list, matters. 
So `[PUTIN] <food_food> (1) <freezer> (1)`  and `[PUTIN] <freezer> (1) <food_food> (1)` are different actions, because the order of parameters has a semantic meaning.
In this case we would consider the second variant, putting a freezer into food, to be nonsensical.

Other approaches which we did not follow could be to discard all parameters, consider them to be a set rather than a list or choose some other metric for deciding if parameters are the same.
 


## Training Process (Augustin)

## Evaluation 

### Clustering

### Semantic Analysis and Embedding Arithmetic

Two of the ways we considered for evaluating the semantic properties of the action embeddings was checking for similarity and evaluating embedding arithmetic.

#### Checking for similar vectors

One easy way of checking if word embeddings manage to find meaningful semantic representations is checking which embeddings are most similar to each other and whether the represented actions are similar.
This follows from the initial underlying assumption of these embeddings, which is that actions or words can be semantically defined by the company they keep, i.e. their context.

When choosing some random embeddings and searching for the most similar ones we did find satisfying results, for example:

```
most similar to Walk, ('mirror',) are ["TurnTo, ('mirror',)", "LookAt, ('mirror',)", "Find, ('mirror',)",...]
most similar to Grab, ('box',) are ["Find, ('box',)", "Write, ('box',)", ... ]
```

However some of the results also left much to be desired:

```
most similar to Walk, ('sauce',) are ["Cut, ('food_kiwi',)", "PutBack, ('pot', 'stove')", "PutBack, ('soap', 'paper_towel')",...]
```

#### Embedding Arithmetic
As mentioned earlier, one function of word embeddings is that semantic relationships can be expressed arithmetically.  
Going back to the example of countries and their capitals it can be shown, that the following roughly holds for word embeddings:

```
Germany - Berlin ≈ France - Paris
```

Rearranging the terms gives:
```
Germany - Berlin + Paris ≈ France
```

So one way of checking for this type of relationship is calculating the right hand side, and finding the embedding which is most similar that result.
The most similar embedding in this case would be the one trained for France.

We tried recreating this with our action embeddings:
```
Find (Glass) - Grab(Glass) ≈ Find(Coffee) - Grab(Coffee)
```
And again rearranging as:
```
Find(Glass) - Grab(Glass) + Grab(Coffee) ≈ Find(Coffee)
```

However, when trying to achieve this result, the most similar vector was always `Find(Glass)` again:

```
"Find, ('glass',)" + "Grab, ('coffee',)" - "Grab, ('glass',)" = ["Find, ('glass',)", "Find, ('kitchen_cabinet',)", "Grab, ('glass',)", "Close, ('kitchen_cabinet',)", "Open, ('kitchen_cabinet',)", "Find, ('cupboard',)", "Open, ('cupboard',)", "Walk, ('kitchen_cabinet',)", "Walk, ('cupboard',)", "Close, ('cupboard',)"]
```

We hoped the properties of 'Grab' and 'Glass' would cancel out, leading to "Find, ('coffee',)". 
As discussed in other parts of this document, there might be many reasons for poor or unexpected performance.  

All of these semantic tests can be run by executing the ```semantics.py``` script.

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
