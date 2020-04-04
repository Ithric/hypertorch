from modules.elevated.models import ElevatedModel, SearchSpace, SearchSpacePrimitives
import torch
from torch import nn

class ElevatedLinear(ElevatedModel):
    def __init__(self, name, n_output_nodes=None):
        super(ElevatedLinear, self).__init__(name)
        self.__name = name
        self.__n_output_nodes = n_output_nodes

    def materialize(self, individual, input_shape):
        self.layer = nn.Linear(input_shape[-1], self.__n_output_nodes or individual["nodes"])
        return self
        
    def forward(self, input_data):
        return self.layer(input_data)

    def get_searchspace(self):
        if self.__n_output_nodes != None: return None
        else:
            return SearchSpace(self.__class__.__name__, self.__name, {
                "nodes" : SearchSpacePrimitives.IntSpace.default_range
            })