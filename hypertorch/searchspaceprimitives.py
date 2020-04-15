
class IntSpace(object):
    def __init__(self, from_int, to_int):
        self.from_int = from_int
        self.to_int = to_int

    def __str__(self):
        return "IntSpace(from_int={}, to_int={})".format(self.from_int, self.to_int)

class FloatSpace(object):
    def __init__(self, from_float, to_float):
        self.from_float = from_float
        self.to_float = to_float

    def __str__(self):
        return "FloatSpace(from_float={}, to_float={})".format(self.from_float, self.to_float)

class NoSpace(object):
    def __init__(self, value):
        self.exact_value = value
    
    def __str__(self):
        return "Exactly({})".format(self.exact_value)