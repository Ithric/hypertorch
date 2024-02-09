import unittest
import hypertorch2
import torch


class TestLayers(unittest.TestCase):
    def test_linear_layer(self):
        hyper_model = hypertorch2.hyperlayers.HyperLinear(13)
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()
        
        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 13))

    def test_dropout_layer(self):
        hyper_model = hypertorch2.hyperlayers.HyperDropout(0.5)
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()
        
        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 6))

    def test_gaussnoise_layer(self):
        hyper_model = hypertorch2.hyperlayers.HyperGaussNoise(0.5)
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()
        
        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 6))

    def test_rnn_layer(self):
        hyper_model = hypertorch2.hyperlayers.HyperRNN(ndims=13)
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()
        
        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 13))

    def test_node_selector(self):
        hyper_model = hypertorch2.hyperlayers.HyperNodeSelector({
            "option_a" : hypertorch2.hyperlayers.HyperLinear(13),
            "option_b" : hypertorch2.hyperlayers.HyperLinear(6)
        }, default_key="option_b")
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()

        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 6))

    def test_sequential(self):
        hyper_model = hypertorch2.hyperlayers.HyperSequential(
            hypertorch2.hyperlayers.HyperLinear(13),
            hypertorch2.hyperlayers.HyperDropout(0.5),
            hypertorch2.hyperlayers.HyperLinear(6)
        )
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()

        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 6))

    def test_hyperlayernorm(self):
        hyper_model = hypertorch2.hyperlayers.HyperLayerNorm()
        searchspace = hyper_model.build_searchspace()
        individual = searchspace.default_individual()

        layer_input = torch.randn(12,4,6)
        hyper_model.materialize(individual, layer_input)
        prediction : torch.Tensor = hyper_model(layer_input)
        self.assertEqual(prediction.shape, (12, 4, 6))


# poetry run python -m tests.layer_tests
if __name__ == '__main__':
    unittest.main()