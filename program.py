import torch 
import pendulum as pm
import numpy as np
import elevated
from elevated.nodelib import *
from common import pyutils
from sklearn import datasets
from functools import reduce, partial

class MyTestModelSubModule(elevated.ElevatedModel):
    def __init__(self, name):
        super(MyTestModelSubModule, self).__init__(name)
        self.hidden_layer = ElevatedLinear(name="sub_linear")

    def forward(self, x):
        return self.hidden_layer(x)

class MyTestModel(elevated.ElevatedModel):
    def __init__(self):
        super(MyTestModel, self).__init__("root")
        # self.layer_a = ElevatedLinear(name="layer_a")
        self.layer_a = MyTestModelSubModule("layer_a")
        self.layer_b = ElevatedLinear("layer_b")
        self.layer_c = ElevatedLinear("layer_c", n_output_nodes=1)
        pass

    def forward(self, x):
        y = self.layer_a(x[0])
        y = self.layer_b(y)
        y = self.layer_c(y)
        return y




inputs = [torch.from_numpy(np.random.uniform(size=(12,4,6)).astype(np.float32))]
elevated_model = MyTestModel()
searchspace = elevated_model.get_searchspace()
individual = searchspace.default_individual()
# individual = elevated.Individual.coalesce(individual, elevated.Individual.parse({
#   "layer_a": {'nodes': 340}
# }))

# materialize and test the model
test_model = elevated_model.materialize(individual, [kx.shape[1:] for kx in inputs])
print("Model materialized successfully")
# print(list(test_model.parameters()))
exit(1)

prediction = test_model(inputs)
print("Prediction:", prediction, prediction.shape)
exit(1)


# Utilities goes here
def model_trainer(data):
    raise NotImplementedError()

def model_evaluator(memento, data):
    raise NotImplementedError()

def individual_evaluator(elevated_model, default_individual, training_data, evaluation_data, individual):

    # Build the model on top of a default_individual coalesced with the optimization individual
    individual = Individual.coalesce(default_individual, existing_individual)
    model = elevated_model.materialize(individual)

    # Test the model and return an optimization signal
    trained_model = model_trainer(training_data)
    predictions = model_evaluator(trained_model, evaluation_data)
    error = calc_error(predictions)
    return error


# existing_individual = {
#     "/root/name_of_node_leg_a" : [{"type" : "linear", "nodes" : 34 }]
# }


# Fake optimizer application

def create_data_by_shapes(n_samples, x_shapes, y_shapes):
    import numpy as np
    return [np.random.uniform(size=(n_samples,) + shape) for shape in x_shapes], [np.random.uniform(size=(n_samples,) + shape) for shape in y_shapes]

x_shapes = [
    ("input_a", (24,))
    ("input_b", (55,2))
    ("input_c", (128,4,3))
]
y_shapes = [(2,),(24,2)]


# Create an elevated model and extract data about the model and searchspace
elevated_model = MyTestModel(dict(x_shapes))
searchspace = elevated_model.get_searchspace()
default_individual = searchspace.default_individual()

# Evaluate a randomly created individual
training_data = create_data_by_shapes(x_shapes.values, y_shapes)
evaluation_data = create_data_by_shapes(x_shapes.values, y_shapes)
evaluator = partial(individual_evaluator, elevated_model, default_individual, training_data, evaluation_data)
loss = evaluator(elevated_model, searchspace.create_random())

print("Loss: ", loss)
