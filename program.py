import torch 
import numpy as np

import hypertorch2 as hypertorch
from hypertorch2.nodelib import *
from hypertorch2.searchspaceprimitives import *

class MyTestModelSubModule(hypertorch.HyperModel):
    def __init__(self):
        super(MyTestModelSubModule, self).__init__(debug_name="MyTestModelSubModule", labels=["test-submodel"])
        self.hidden_layer = HyperLinear()

    def forward(self, x : dict, test_scalar):
        assert isinstance(x, dict), "x should be a dict, but its not!"
        return self.hidden_layer(x["test"])*test_scalar

class MyTestModel(hypertorch.HyperModel):
    def __init__(self):
        super(MyTestModel, self).__init__(debug_name="MyTestModel", labels=["test"])
        # self.layer_a = HyperLinear(name="layer_a")
        self.layer_dict = nn.ModuleDict({
            "layer_a" : HyperLinear(),
            "layer_b" : MyTestModelSubModule()
        })
        self.layer_list = nn.ModuleList([
            HyperLinear(),
            HyperLinear()
        ])
        self.hypersequential = HyperSequential(
            HyperLinear(),
            torch.nn.ReLU(),
            HyperLinear()
        )
        self.layer_c = HyperLinear(n_output_nodes=1)
        self.activation_function = HyperNodeSelector({
            "elu" : torch.nn.ELU(),
            "relu" : torch.nn.ReLU(),
        }, default_key="elu")
        pass

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        # TODO: use the y tensor as well
        y = self.layer_dict["layer_a"](x)
        y = self.layer_dict["layer_b"]({ "test" : y, "fakeerr": [], "fakeerr2" : None}, test_scalar=1.1)
        y = self.layer_c(y)
        for layer in self.layer_list:
            y = layer(y)
        y = self.hypersequential(y)
        y = self.activation_function(y)
        return y


inputs = [
    torch.from_numpy(np.random.uniform(size=(12,4,6)).astype(np.float32)),
    torch.from_numpy(np.random.uniform(size=(12,3)).astype(np.float32))
]
hyper_model = MyTestModel()

# Build the searchspace and create an individual
extraspace = {
    "HyperLinear" : { "nodes" : hypertorch.searchspaceprimitives.IntSpace(1,350) }
}
searchspace = hyper_model.build_searchspace(default_layer_searchspace={**hypertorch.DefaultLayerSpace, **extraspace} )
individual = searchspace.default_individual()

# # materialize and test the model
hyper_model.materialize(individual, inputs[0], inputs[1])

prediction = hyper_model(inputs[0], inputs[1])
print("Model prediction shape:", prediction.shape)

