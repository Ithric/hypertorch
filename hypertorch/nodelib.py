from hypertorch.models import HyperModel, SearchSpace, NullSpace
from hypertorch.searchspaceprimitives import *
import torch
from torch import nn

def auto_space(value):
    if isinstance(value, (float,int)):
        return NoSpace(value)
    else:
        return value

class HyperLinear(HyperModel):
    def __init__(self, name, n_output_nodes=None):
        super(HyperLinear, self).__init__(name)
        self.__searchspace = SearchSpace(self.__class__.__name__, self.Name, {
            "nodes" : auto_space(n_output_nodes)
        }) 

    def materialize(self, individual, input_shape):
        self.layer = nn.Linear(input_shape[-1], individual["nodes"])
        return self
        
    def forward(self, input_data):
        return self.layer(input_data)

    def get_searchspace(self):
        return self.__searchspace

class HyperDropout(HyperModel):
    def __init__(self, name):
        super(HyperDropout, self).__init__(name)
        self.__searchspace = SearchSpace(self.__class__.__name__, self.Name, {
            "dropout_p" : None
        })
    
    def materialize(self, individual, input_shapes, torch_module_list=[]):
        self.layer = nn.Dropout(individual["dropout_p"])
        return self

    def forward(self, x):
        return self.layer(x)

    def get_searchspace(self):
        return self.__searchspace


class HyperGaussNoise(HyperModel):
    def __init__(self, name, sigma=None):
        super(HyperGaussNoise, self).__init__(name)
        self.__noise = torch.tensor(0,)
        self.__searchspace = SearchSpace(self.__class__.__name__, self.Name, {
            "sigma" : sigma
        })
        
    def materialize(self, individual, input_shapes, torch_module_list=[]):
        self.__sigma = individual["sigma"]
        return self

    def forward(self, x):
        self.__noise = self.__noise.to(x.device)
        if self.training and self.__sigma != 0:
            sampled_noise = self.__noise.repeat(*x.size()).float().normal_() * (self.__sigma * x)
            x = x + sampled_noise
        
        return x 
        
    def get_searchspace(self):
        return self.__searchspace

class HyperNoOp(HyperModel):
    def __init__(self, name):
        super(HyperNoOp, self).__init__(name)
    
    def materialize(self, individual, input_shapes, torch_module_list=[]):
        return self

    def forward(self, x):
        return x 
        
    def get_searchspace(self):
        return SearchSpace("NoOp", "NoOp")

class HyperNodeSelector(HyperModel):
    def __init__(self, name, hyperNodes : dict, default_key):
        super(HyperNodeSelector, self).__init__(name)
        self.__hyperNodes = hyperNodes

        conditional_spaces = SearchSpace("ConditionalNode","spaces")
        for hyperNodeKey,hyperNode in hyperNodes.items():
            conditional_spaces.append_child(hyperNodeKey, hyperNode.get_searchspace())

        self.__searchspace = SearchSpace(self.__class__.__name__, self.Name, {
            "key" : OneOfSet([k for k in hyperNodes.keys()]),
            "spaces" : conditional_spaces
        })
        self.__default_key = default_key

    def materialize(self, individual, input_shapes, torch_module_list=[]):
        selected_key = individual["key"]
        if selected_key == "default_key": selected_key = self.__default_key
        self.active_node = self.__hyperNodes[selected_key].materialize(individual["spaces"][selected_key], input_shapes)
        return self

    def forward(self, x):
        return self.active_node.forward(x)
        
    def get_searchspace(self):
        return self.__searchspace