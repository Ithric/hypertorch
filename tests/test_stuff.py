import unittest
import hypertorch2
import torch

class ListWrapperModel(hypertorch2.HyperModel):
    def __init__(self, layer : torch.nn.ModuleList):
        super(ListWrapperModel, self).__init__(debug_name="ListWrapperModel", labels=["test"])
        self.layer = layer

    def forward(self, x : torch.Tensor):
        for layer in self.layer:
            x = layer(x)
        return x
        

class TestContainers(unittest.TestCase):   
    def test_nn_list(self):
        reuse_layer = hypertorch2.hyperlayers.HyperLinear(6)
        hyper_model = torch.nn.ModuleList([
            reuse_layer,
            reuse_layer,
            hypertorch2.hyperlayers.HyperLinear(7)
        ])
        hyper_model = ListWrapperModel(hyper_model)
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()

        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 7))


# poetry run python -m tests.test_stuff
if __name__ == '__main__':
    unittest.main()