from typing import List, Dict, Any, Union, Optional, Set

class SearchSpacePrimitive(object):
    pass

class IntSpace(SearchSpacePrimitive):
    def __init__(self, from_int : int, to_int : int, default : Optional[int] = None):
        self.from_int = from_int
        self.to_int = to_int
        self.default = default

    def __str__(self):
        return "IntSpace(from_int={}, to_int={})".format(self.from_int, self.to_int)

    def __repr__(self) -> str:
        return self.__str__()

class FloatSpace(SearchSpacePrimitive):
    def __init__(self, from_float : float, to_float : float, default : Optional[float] = None):
        self.from_float = from_float
        self.to_float = to_float
        self.default = default

    def __str__(self):
        return "FloatSpace(from_float={}, to_float={})".format(self.from_float, self.to_float)

    def __repr__(self) -> str:
        return self.__str__()

class NoSpace(SearchSpacePrimitive):
    def __init__(self, value : Any):
        self.exact_value = value
    
    def __str__(self):
        return "Exactly({})".format(self.exact_value)

    def __repr__(self) -> str:
        return self.__str__()

class OneOfSet(SearchSpacePrimitive):
    def __init__(self, set_values : List[Any], default : Optional[List[Any]] = None):
        self.set_values = [k for k in set_values]
        self.default = default
    
    def __str__(self):
        return "OneOfSet({})".format(self.set_values)

    def __repr__(self) -> str:
        return self.__str__()

