class Action:
    def __init__(self, action: str, targets: [str]):
        self.action = action
        self.targets = tuple(targets)

    def __eq__(self, other):
        return self.action == other.action and self.targets == other.targets

    def __hash__(self) -> int:
        return hash((self.action, self.targets))

    def __str__(self):
        return f"{self.action}, {self.targets}"
