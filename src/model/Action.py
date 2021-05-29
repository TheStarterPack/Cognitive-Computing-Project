class Action:
    def __init__(self, action, targets):
        self.action = action
        self.targets = tuple(targets)

    def __eq__(self, other):
        return self.action == other.action and self.targets == other.targets

    def __key(self):
        return self.action, self.targets

    def __hash__(self):
        return hash(self.__key())
