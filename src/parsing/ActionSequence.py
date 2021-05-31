from .Action import Action
import numpy as np

class ActionSequence:

    def __init__(self, actions: [Action]):
        self.actions = actions

    def __iter__(self):
        return iter(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, item):
        return self.actions[item]

def generate_contexts(action_sequences: [ActionSequence], context_length: int = 2):
    assert context_length >= 1

    contexts = []
    centers = []
    for sequence in action_sequences:
        assert isinstance(sequence, ActionSequence)
        for i in range(context_length, len(sequence) - context_length - 1):
            right_context = sequence[i - context_length: i]
            left_context = sequence[i + 1: i + context_length + 1]
            context = right_context + left_context
            center = sequence[i]

            assert isinstance(center, Action)
            assert len(right_context) == len(left_context)
            assert len(context) == context_length * 2
            assert right_context + [center] + left_context == \
                   sequence[i - context_length: i + context_length + 1]

            contexts.append(context)

            centers.append([center])

    return contexts, centers


def actions_to_tokenized_np_arrays(action_lists: [[Action]], action_to_id: dict):
    tokenized_actions = []
    for actions in action_lists:
        np_array = np.fromiter((action_to_id[action] for action in actions), dtype=np.int64)
        tokenized_actions.append(np_array)

    return np.stack(tokenized_actions)