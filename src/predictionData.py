
import jsonpickle
import numpy as np
import random

from parsing.actionSequence import ActionSequence, generate_contexts
from parsing.parser import ActionSeqParser

def get_prediction_data(approach_name, context_size):
    #read embedding dictionary
    json_string = ''
    with open(f'embeddings/{approach_name}'.json) as file:
        json_string = file.read()
    embedding_dict = jsonpickle.decode(json_string)

    #get action contexts
    action_sequences = ActionSeqParser.read_action_seq_corpus()
    (contexts, centers) = generate_contexts(action_sequences, context_size)

    #get embeddings for contexts, encode centers as one-hot target vectors
    data = []
    for i in range(len(contexts)):
        context = contexts[i]
        center = centers[i]
        context_embedded = []
        target = np.zeros(len(embedding_dict))

        for context_action in context:
            context_embedded.append(embedding_dict[context_action][1])
        
        onehot_idx = embedding_dict[center][0]
        target[onehot_idx] = 1

        data.append(context_embedded, target)

    #split data into train and test
    random.shuffle(data)
    l = int(len(data) * 0.1)
    test = data[0:l]
    train = data[l:]
    
    return train, test