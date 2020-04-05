from elevated.models import ElevatedModel, SearchSpace, SearchSpacePrimitives
import torch
from torch import nn

class ElevatedLinear(ElevatedModel):
    def __init__(self, name, n_output_nodes=None):
        super(ElevatedLinear, self).__init__(name)
        self.__n_output_nodes = n_output_nodes

    def materialize(self, individual, input_shape):
        self.layer = nn.Linear(input_shape[-1], self.__n_output_nodes or individual["nodes"])
        return self
        
    def forward(self, input_data):
        return self.layer(input_data)

    def get_searchspace(self):
        if self.__n_output_nodes != None: return None
        else:
            return SearchSpace(self.__class__.__name__, self.Name, {
                "nodes" : SearchSpacePrimitives.IntSpace.default_range
            })

class ElevatedDropout(ElevatedModel):
    def __init__(self, name):
        super(ElevatedDropout, self).__init__(name)
    
    def materialize(self, individual, input_shapes, torch_module_list=[]):
        self.layer = nn.Dropout(individual["dropout_p"])
        return self

    def forward(self, x):
        return self.layer(x)

    def get_searchspace(self):
        return SearchSpace(self.__class__.__name__, self.Name, {
            "dropout_p" : SearchSpacePrimitives.FloatSpace(0.0, 1.0)
        })


class GaussianNoise(ElevatedModel):
    def __init__(self, name, sigma=0.1):
        super(GaussianNoise, self).__init__(name)
        self.sigma = sigma
        self.noise = torch.tensor(0,)
        self.is_bound_to_device = False

    def forward(self, x):
        self.noise = self.noise.to(x.device)
        if self.training and self.sigma != 0:
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * (self.sigma * x)
            x = x + sampled_noise
        
        return x 
        
    def get_searchspace(self):
        return SearchSpace(self.__class__.__name__, self.Name, {
            "sigma" : SearchSpacePrimitives.FloatSpace(0.0, 1.0)
        })

