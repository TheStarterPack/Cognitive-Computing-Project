import os
import re
from pathlib import Path

from .Action import Action
from .ActionSequence import ActionSequence

AUGMENTED_PATHS = [Path("augmented/augment_exception/withoutconds"),
                   Path("augmented/augment_location/withoutconds")]
DEFAULT_PATHS = [Path("programs_processed_precond_nograb_morepreconds/withoutconds")]


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
            full_path = str(path)

            for root, _, files in os.walk(full_path):
                for filename in files:
                    with open(os.path.join(root, filename)) as file:
                        action_sequence = []
                        actions = map(lambda x: x.strip(), file.readlines()[4:])
                        for action in actions:
                            title = re.findall("\[(.+?)]", action)
                            targets = re.findall("<(.+?)>", action)
                            if title:
                                action_sequence.append(Action(title[0], targets))
                        action_sequences.append(ActionSequence(action_sequence))
        self.action_sequences = action_sequences
        return action_sequences

    def get_action_to_id_dict(self):
        unique_actions = {action for seq in self.action_sequences for action in seq.actions}
        return {k: v for k, v in zip(unique_actions, range(len(unique_actions)))}
