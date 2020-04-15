from hypertorch.models import HyperModel, SearchSpace
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

