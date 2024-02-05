import torch 
import numpy as np

import hypertorch
from hypertorch.nodelib import *
from hypertorch.searchspaceprimitives import *

class BasicModuleTest(torch.nn.Module):
    def __init__(self):
        super(BasicModuleTest,self).__init__()
        self.layer = torch.nn.Linear(64,64)
        pass

    def forward(self,x):
        assert len(x) == 1, "x is wrong length"
        return self.layer(x[0])

class MyTestModelSubModule(hypertorch.HyperModel):
    def __init__(self):
        super(MyTestModelSubModule, self).__init__()
        self.hidden_layer = HyperLinear()

    def forward(self, x : dict, test_scalar):
        assert isinstance(x, dict), "x should be a dict, but its not!"
        return self.hidden_layer(x["test"])*test_scalar

class MyTestModel(hypertorch.HyperModel):
    def __init__(self):
        super(MyTestModel, self).__init__(debug_name="MyTestModel")
        # self.layer_a = HyperLinear(name="layer_a")
        self.layer_a = HyperLinear()
        self.layer_b = MyTestModelSubModule()
        self.layer_bb = BasicModuleTest()
        self.layer_c = HyperLinear(n_output_nodes=1)
        pass

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        # TODO: use the y tensor as well
        print("im here")
        y = self.layer_a(x)
        y = self.layer_b({ "test" : y, "fakeerr": [], "fakeerr2" : None}, test_scalar=1.1)
        y = self.layer_bb([y])
        y = self.layer_c(y)
        print("now im here")
        return y




inputs = [
    torch.from_numpy(np.random.uniform(size=(12,4,6)).astype(np.float32)).to("cuda"),
    torch.from_numpy(np.random.uniform(size=(12,3)).astype(np.float32)).to("cuda")
]
hyper_model = MyTestModel()

extraspace = {
    "HyperLinear" : { "nodes" : hypertorch.searchspaceprimitives.IntSpace(1,350) }
}
searchspace = hyper_model.get_searchspace(default_layer_searchspace={**hypertorch.DefaultLayerSpace, **extraspace} )
individual = searchspace.default_individual()

# individual = hyper.Individual.coalesce(individual, hyper.Individual.parse({
#   "layer_a": {'nodes': 340}
# }))

# materialize and test the model
test_model = hyper_model.materialize(individual, hypertorch.ExpandedArg([kx.shape[1:] for kx in inputs]))
test_model = test_model.to("cuda")

prediction = test_model(inputs[0], inputs[1])
print("Prediction:", prediction, prediction.shape)


