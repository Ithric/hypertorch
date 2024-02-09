from .models import HyperModel, SearchSpace, NullSpace, MaterializationContext, Individual
from .searchspaceprimitives import *
import torch
from torch import nn

def auto_space(value):
    if isinstance(value, (str,float,int)):
        return NoSpace(value)
    else:
        return value


class HyperLinear(HyperModel):
    def __init__(self, n_output_nodes : int = None):
        super(HyperLinear, self).__init__()
        self.n_output_nodes = auto_space(n_output_nodes)

    def materializing_forward(self, ctx : MaterializationContext, orig_forward, x : torch.Tensor):
        self.linear = torch.nn.Linear(x.shape[-1], ctx.individual["nodes"])
        return orig_forward(x)
    
    def forward(self, x : torch.Tensor):
        return self.linear(x)

    def get_searchspace(self, default_space : Dict[str,Any]) -> SearchSpace:
        return SearchSpace(self.__class__.__name__, {
            "nodes" : self.n_output_nodes or default_space["nodes"]
        }) 

class HyperDropout(HyperModel):
    def __init__(self, p = None):
        super(HyperDropout, self).__init__()
        self.__dropout = auto_space(p)
            
    def materializing_forward(self, ctx : MaterializationContext, orig_forward, x : torch.Tensor):
        self.layer = torch.nn.Dropout(ctx.individual["dropout_p"])
        return orig_forward(x)

    def forward(self, x):
        return self.layer(x)

    def get_searchspace(self, default_space : Dict[str,Any]) -> SearchSpace:
        return SearchSpace(self.__class__.__name__, {
            "dropout_p" : self.__dropout or default_space["dropout_p"]
        })



class HyperGaussNoise(HyperModel):
    def __init__(self, sigma_space=None):
        super(HyperGaussNoise, self).__init__()
        self.__noise = torch.tensor(0,)
        self.__sigma_space = auto_space(sigma_space)
        
    def materializing_forward(self, ctx : MaterializationContext, orig_forward, x : torch.Tensor):
        self.__sigma = ctx.individual["sigma"]
        return orig_forward(x)

    def forward(self, x):
        self.__noise = self.__noise.to(x.device)
        if self.training and self.__sigma != 0:
            sampled_noise = self.__noise.repeat(*x.size()).float().normal_() * (self.__sigma * x)
            x = x + sampled_noise
        
        return x 
        
    def get_searchspace(self, default_space : Dict[str,Any]) -> SearchSpace:
        return SearchSpace(self.__class__.__name__, {
            "sigma" : self.__sigma_space or default_space["sigma"]
        })

class HyperNoOp(HyperModel):
    def __init__(self):
        super(HyperNoOp, self).__init__()
    
    def materializing_forward(self, ctx : MaterializationContext, orig_forward, x : torch.Tensor):
        return orig_forward(x)

    def forward(self, x):
        return x 
        
    def get_searchspace(self, default_space : Dict[str,Any]) -> SearchSpace:
        return SearchSpace(self.__class__.__name__, {})
    

class HyperRNN(HyperModel):
    def __init__(self, mode=None, ndims=None, bidir=None, n_layers=None, return_sequences=True, **kwargs):
        super(HyperRNN, self).__init__()
        self.__return_sequences = return_sequences
        self.__searchspace_dict = {
            "mode" : auto_space(mode or OneOfSet(["GRU","LSTM"])),
            "ndims" : auto_space(ndims or IntSpace(1,256)),
            "bidir" : auto_space(bidir or OneOfSet([True,False])),
            "nlayers" : auto_space(n_layers or IntSpace(1,5))
        }

    def materializing_forward(self, ctx : MaterializationContext, orig_forward, x : torch.Tensor):
        mode = ctx.individual["mode"]
        if mode == "GRU":
            self.rnn = torch.nn.GRU(x.shape[-1], ctx.individual["ndims"], num_layers=ctx.individual["nlayers"], bidirectional=ctx.individual["bidir"], batch_first=True)
        elif mode == "LSTM":
            self.rnn = torch.nn.LSTM(x.shape[-1], ctx.individual["ndims"], num_layers=ctx.individual["nlayers"], bidirectional=ctx.individual["bidir"], batch_first=True)
        else:
            raise Exception("Unknown RNN type: {}".format(mode))
        
        return orig_forward(x)
        
    def forward(self, x):
        lstm_output, state = self.rnn(x)
        if self.__return_sequences:
            return lstm_output
        else:
            return lstm_output[:,-1].contiguous()

    def get_searchspace(self, default_space : Dict[str,Any]) -> SearchSpace:
        coallesced_space = {
            "mode" : self.__searchspace_dict["mode"] or default_space["mode"],
            "ndims" : self.__searchspace_dict["ndims"] or default_space["ndims"],
            "bidir" : self.__searchspace_dict["bidir"] or default_space["bidir"],
            "nlayers" : self.__searchspace_dict["nlayers"] or default_space["nlayers"]
        }
        return SearchSpace(self.__class__.__name__, coallesced_space) 


class HyperNodeSelector(HyperModel):
    def __init__(self, hyperNodes : Dict[str,HyperModel], default_key : str):
        super(HyperNodeSelector, self).__init__()
        self.__hyperNodes = hyperNodes
        self.__default_key = default_key

    def materializing_forward(self, ctx : MaterializationContext, orig_forward, *args, **kwargs):
        selected_key = ctx.individual["key"]
        if selected_key == "default_key": selected_key = self.__default_key

        selected_layer = self.__hyperNodes[selected_key]
        if isinstance(selected_layer, HyperModel):
            active_node_individual = Individual(ctx.individual["spaces"][selected_key])
            selected_layer.materialize(active_node_individual, *args, **kwargs)
            self.active_node = selected_layer
        else:
            self.active_node = self.__hyperNodes[selected_key]
        return orig_forward(*args, **kwargs)

    def forward(self, x):
        return self.active_node.forward(x)
        
    def get_searchspace(self, default_space : Dict[str,Any]) -> SearchSpace:
        conditional_spaces = SearchSpace("ConditionalNode")
        for hyperNodeKey,hyperNode in self.__hyperNodes.items():
            if isinstance(hyperNode, HyperModel):
                conditional_spaces.append_child(hyperNodeKey, hyperNode.build_searchspace()) # TODO: bad space! this will discard the supplied searchspace in favour of defaultspace
            else:
                continue

        return SearchSpace(self.__class__.__name__, {
            "key" : OneOfSet([k for k in self.__hyperNodes.keys()]),
            "spaces" : conditional_spaces
        })


class HyperSequential(HyperModel):
    def __init__(self, *submodels):
        super(HyperSequential, self).__init__()
        self.submodels = torch.nn.ModuleList(submodels)

    def forward(self, x):
        for layer in self.submodels:
            x = layer(x)
        return x
    



class HyperLayerNorm(HyperModel):
    def __init__(self):
        super(HyperLayerNorm, self).__init__()
        pass

    def materializing_forward(self, ctx : MaterializationContext, orig_forward, x : torch.Tensor):
        self.layer = torch.nn.LayerNorm(x.shape[-1])
        return orig_forward(x)

    def forward(self, x):
        return self.layer(x)