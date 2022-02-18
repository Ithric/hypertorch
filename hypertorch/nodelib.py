from hypertorch.models import HyperModel, SearchSpace, NullSpace, MaterializedModel
from hypertorch.searchspaceprimitives import *
import torch
from torch import nn

def auto_space(value):
    if isinstance(value, (str,float,int)):
        return NoSpace(value)
    else:
        return value

class HyperLinear(HyperModel):
    def __init__(self, n_output_nodes=None):
        super(HyperLinear, self).__init__()
        self.__n_output_nodes = n_output_nodes

    def materialize(self, individual, input_shape, **kwargs):
        self.layer = nn.Linear(input_shape[-1], individual["nodes"])
        return self.layer
        
    def forward(self, input_data):
        return self.layer(input_data)

    def get_searchspace(self, **kwargs):
        return SearchSpace(self.__class__.__name__, {
            "nodes" : auto_space(self.__n_output_nodes)
        }) 

class HyperDropout(HyperModel):
    def __init__(self, p = None):
        super(HyperDropout, self).__init__()
        self.__dropout = auto_space(p)
            
    def materialize(self, individual, input_shapes, torch_module_list=None, **kwargs):
        self.layer = nn.Dropout(individual["dropout_p"])
        return self.layer

    def forward(self, x):
        return self.layer(x)

    def get_searchspace(self, **kwargs):
        return SearchSpace(self.__class__.__name__, {
            "dropout_p" : self.__dropout
        })



class HyperGaussNoise(HyperModel):
    def __init__(self, sigma_space=None):
        super(HyperGaussNoise, self).__init__()
        self.__noise = torch.tensor(0,)
        self.__sigma_space = auto_space(sigma_space)
        
    def materialize(self, individual, input_shapes, torch_module_list=None, **kwargs):
        self.__sigma = individual["sigma"]
        return MaterializedModel(self, [])

    def forward(self, x):
        self.__noise = self.__noise.to(x.device)
        if self.training and self.__sigma != 0:
            sampled_noise = self.__noise.repeat(*x.size()).float().normal_() * (self.__sigma * x)
            x = x + sampled_noise
        
        return x 
        
    def get_searchspace(self, **kwargs):
        return SearchSpace(self.__class__.__name__, {
            "sigma" : self.__sigma_space
        })

class HyperNoOp(HyperModel):
    def __init__(self):
        super(HyperNoOp, self).__init__()
    
    def materialize(self, individual, input_shapes, torch_module_list=None, **kwargs):
        return MaterializedModel(self, [])

    def forward(self, x):
        return x 
        
    def get_searchspace(self, **kwargs):
        return SearchSpace("NoOp")

class HyperNodeSelector(HyperModel):
    def __init__(self, hyperNodes : dict, default_key):
        super(HyperNodeSelector, self).__init__()
        self.__hyperNodes = hyperNodes

        conditional_spaces = SearchSpace("ConditionalNode")
        for hyperNodeKey,hyperNode in hyperNodes.items():
            if isinstance(hyperNode, HyperModel):
                conditional_spaces.append_child(hyperNodeKey, hyperNode.get_searchspace())
            else:
                continue

        self.__default_key = default_key
        self.__searchspace_dict =  {
            "key" : OneOfSet([k for k in hyperNodes.keys()]),
            "spaces" : conditional_spaces
        }

    def materialize(self, individual, input_shapes, torch_module_list=None, **kwargs):
        selected_key = individual["key"]
        if selected_key == "default_key": selected_key = self.__default_key
        if isinstance(self.__hyperNodes[selected_key], HyperModel):
            self.active_node = self.__hyperNodes[selected_key].materialize(individual["spaces"][selected_key], input_shapes)
            assert isinstance(self.active_node, torch.nn.Module), "Materialization failed of node with key: {}: Got {}".format(selected_key, self.active_node)
        else:
            self.active_node = self.__hyperNodes[selected_key]
        return self.active_node

    def forward(self, x):
        return self.active_node.forward(x)
        
    def get_searchspace(self, **kwargs):
        return SearchSpace(self.__class__.__name__, self.__searchspace_dict)

