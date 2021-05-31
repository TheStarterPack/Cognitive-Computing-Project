import os
import re

from src.model.Action import Action
from src.model.ActionSequence import ActionSequence

AUGMENTED_PATHS = ["augmented/augment_exception/withoutconds",
                   "augmented/augment_location/withoutconds"]
DEFAULT_PATHS = ["programs_processed_precond_nograb_morepreconds/withoutconds"]

ROOT_PATH = os.path.curdir + "/../"


class ActionSeqParser:

    def __init__(self, include_augmented: bool, include_default: bool):
        paths = []
        if include_augmented:
            paths += AUGMENTED_PATHS
        if include_default:
            paths += DEFAULT_PATHS
        self.paths = paths
        self.action_sequences = []

    def read_action_seq_corpus(self):
        action_sequences = []
        for path in self.paths:
            full_path = ROOT_PATH + path

            for root, subdirs, files in os.walk(full_path):
                for filename in files:
                    with open(os.path.join(root, filename)) as file:
                        action_sequence = []
                        actions = map(lambda x: x.strip(), file.readlines()[4:])
                        for action in actions:
                            title = re.findall("\[(.+?)\]", action)
                            targets = re.findall("<(.+?)>", action)
                            if title:
                                action_sequence.append(Action(title[0], targets))
                        action_sequences.append(ActionSequence(action_sequence))
        self.action_sequences = action_sequences
        return action_sequences

    def get_tokenization(self):
        unique_actions = {action for seq in self.action_sequences for action in seq.actions}
        return dict(zip(range(len(unique_actions)), unique_actions))
