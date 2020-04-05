import torch
import numpy as np
import elevated
from elevated.nodelib import *
from common import pyutils
from functools import partial

OPTIMIZATION_TRIALS = 25

class MyTestModelSubModule(elevated.ElevatedModel):
    def __init__(self, name):
        super(MyTestModelSubModule, self).__init__(name)
        self.hidden_layer = ElevatedLinear(name="sub_linear")

    def forward(self, x):
        return self.hidden_layer(x)

class MyTestModel(elevated.ElevatedModel):
    def __init__(self):
        super(MyTestModel, self).__init__("root")
        self.layer_a = MyTestModelSubModule("layer_a")
        self.layer_b = ElevatedLinear(name="layer_b")
        self.layer_c = ElevatedLinear(name="layer_c", n_output_nodes=3)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        pass

    def forward(self, x):
        y = self.layer_a(x[0])
        y = self.layer_b(y)
        y = self.relu(y)
        y = self.layer_c(y)
        y = self.softmax(y)
        return [y]


def optimizer_func(evaluations, random_individual_builder, evaluator):
    """
    Arguments:
        evaluations: Number of evaluations before giving up
        random_individual_builder: Creates a random individual. Function signature: def() -> Individual 
        evaluator: def(individual) -> loss value
    Returns: best individual for the job
    """
    results = []
    for epoch in range(evaluations):
        print("Executing evaluation: {} / {}".format(epoch+1, evaluations))
        individual = random_individual_builder()
        loss = evaluator(individual)
        results.append((individual,loss))

    individuals, loss_values  = zip(*results)
    best_index = np.argsort(loss_values)[0]
    return individuals[best_index], loss_values[best_index]


def evaluator(elevated_model, training_data, individual):
    x,y = training_data
    x,x_valid = pyutils.split_data(x, split_factor=0.1)
    y,y_valid = pyutils.split_data(y, split_factor=0.1)
  
    # Materialize the model
    model = elevated_model.materialize(individual, [kx.shape[1:] for kx in x])
    model = pyutils.train_model(model, training_data, validation_data, epochs=20, patience=25, verbosity=0)
    prediction = model(list(map(torch.from_numpy, x_valid)))

    xloss = pyutils.cross_entropy(prediction[0].detach().numpy(), y_valid)
    print(individual)
    print("xloss", xloss)

    del model
    torch.cuda.empty_cache()
    return xloss

# Create the hypermodel
inputs = [torch.from_numpy(np.random.uniform(size=(12,4,6)).astype(np.float32))]
elevated_model = MyTestModel()
searchspace = elevated_model.get_searchspace()

# Run the optimization
training_data, validation_data = pyutils.load_iris_dataset()
best_individual, best_loss = optimizer_func(OPTIMIZATION_TRIALS, searchspace.create_random, partial(evaluator,elevated_model,training_data))
print("Best individual found (loss={}): {}".format(best_loss, best_individual))

# Test the best individual
best_model = elevated_model.materialize(best_individual, [kx.shape[1:] for kx in validation_data[0]])
best_model = pyutils.train_model(best_model, training_data, validation_data, epochs=5000, patience=25, verbosity=1)
predictions = best_model(list(map(torch.from_numpy, validation_data[0])))[0].detach().numpy()

# Calculate accuracy
predictions_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(validation_data[1][0], axis=1)
corect_labels = predictions_labels[predictions_labels == actual_labels]
print("Total accuracy: {}%".format(100 * len(corect_labels) / len(predictions_labels)))