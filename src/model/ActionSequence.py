from src.model.Action import Action


class ActionSequence:

    def __init__(self, actions: [Action]):
        self.actions = actions

    def __iter__(self):
        return iter(self.actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, item):
        return self.actions[item]
