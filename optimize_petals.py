import torch
from sklearn import datasets
import numpy as np
import elevated
from elevated.nodelib import *
from common import pyutils
from functools import partial


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
    Evaluations: Number of evaluations before giving up
    Evaluator: def(individual) -> loss
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


inputs = [torch.from_numpy(np.random.uniform(size=(12,4,6)).astype(np.float32))]
elevated_model = MyTestModel()
searchspace = elevated_model.get_searchspace()

def load_iris_dataset(validation_split_idx=25):
    from sklearn.preprocessing import label_binarize
    from sklearn.utils import shuffle

    iris = datasets.load_iris()
    x = [iris.data[:, :4]] 
    y = [label_binarize(iris.target, classes=[0,1,2])]

    shuffle_index = np.arange(len(x[0]))
    np.random.shuffle(shuffle_index)
    x = [kx[shuffle_index] for kx in x]
    y = [ky[shuffle_index] for ky in y]

    x,x_valid = [kx[:-validation_split_idx].astype(np.float32) for kx in x], [kx[-validation_split_idx:].astype(np.float32) for kx in x]
    y,y_valid = [ky[:-validation_split_idx].astype(np.float32) for ky in y], [ky[-validation_split_idx:].astype(np.float32) for ky in y]
    return (x,y), (x_valid,y_valid)


def split_data(vecs, split_factor=0.1):
    num_samples = set([len(k) for k in vecs])
    assert len(num_samples) == 1, "All vectors must be of equal length"
    num_samples = list(num_samples)[0]
    split_idx = num_samples - int(num_samples*split_factor)

    return [t[:split_idx] for t in vecs], [t[split_idx:] for t in vecs]

def evaluator(training_data, individual):
    x,y = training_data
    x,x_valid = split_data(x, split_factor=0.1)
    y,y_valid = split_data(y, split_factor=0.1)
  
    # Materialize the model
    model = elevated_model.materialize(individual, [kx.shape[1:] for kx in x])
    model = pyutils.train_model(model, training_data, validation_data, epochs=150, verbosity=0)
    prediction = model(list(map(torch.from_numpy, x_valid)))

    xloss = pyutils.cross_entropy(prediction[0].detach().numpy(), y_valid)
    print(individual)
    print("xloss", xloss)

    del model
    torch.cuda.empty_cache()
    return xloss

# Run the optimization
training_data, validation_data = load_iris_dataset()
best_individual, best_loss = optimizer_func(25, searchspace.create_random, partial(evaluator,training_data))

print("Best individual found (loss={}): {}".format(best_loss, best_individual))

best_model = elevated_model.materialize(best_individual, [kx.shape[1:] for kx in validation_data[0]])
best_model = pyutils.train_model(best_model, training_data, validation_data, epochs=150, verbosity=0)
predictions = best_model(list(map(torch.from_numpy, validation_data[0])))[0].detach().numpy()

# Calculate accuracy
predictions_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(validation_data[1][0], axis=1)
corect_labels = predictions_labels[predictions_labels == actual_labels]
print("Total accuracy: {}%".format(100 * len(corect_labels) / len(predictions_labels)))