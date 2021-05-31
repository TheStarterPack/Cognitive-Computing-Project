class Action:
    def __init__(self, action: str, targets: [str]):
        self.action = action
        self.targets = tuple(targets)

    def __eq__(self, other):
        return self.action == other.action and self.targets == other.targets

    def __key(self):
        return self.action, self.targets

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return f"Action: {self.action} Target: {self.targets}"

    def __repr__(self):
        return f"Action: {self.action} Target: {self.targets}"
