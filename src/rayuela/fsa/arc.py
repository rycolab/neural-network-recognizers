class Arc:
    def __init__(self, value, target):
        self._sym = value
        self._target = target

    @property
    def value(self):
        return self._sym

    @property
    def target(self):
        return self._target

    def __iter__(self):
        return iter(tuple((self.value, self.target)))

    def __hash__(self):
        return hash((self.target, self.value))

    def __eq__(self, other):
        return self.target == other.target and self.value == other.value

    def __repr__(self):
        return str(tuple((self.value, self.target)))
