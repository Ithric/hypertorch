
# # Utilities goes here
# def model_trainer(data):
#     raise NotImplementedError()

# def model_evaluator(memento, data):
#     raise NotImplementedError()

# def individual_evaluator(hyper_model, default_individual, training_data, evaluation_data, individual):

#     # Build the model on top of a default_individual coalesced with the optimization individual
#     individual = Individual.coalesce(default_individual, existing_individual)
#     model = hyper_model.materialize(individual)

#     # Test the model and return an optimization signal
#     trained_model = model_trainer(training_data)
#     predictions = model_evaluator(trained_model, evaluation_data)
#     error = calc_error(predictions)
#     return error


# # existing_individual = {
# #     "/root/name_of_node_leg_a" : [{"type" : "linear", "nodes" : 34 }]
# # }


# # Fake optimizer application

# def create_data_by_shapes(n_samples, x_shapes, y_shapes):
#     import numpy as np
#     return [np.random.uniform(size=(n_samples,) + shape) for shape in x_shapes], [np.random.uniform(size=(n_samples,) + shape) for shape in y_shapes]

# x_shapes = [
#     ("input_a", (24,))
#     ("input_b", (55,2))
#     ("input_c", (128,4,3))
# ]
# y_shapes = [(2,),(24,2)]


# # Create an hyper model and extract data about the model and searchspace
# hyper_model = MyTestModel(dict(x_shapes))
# searchspace = hyper_model.get_searchspace()
# default_individual = searchspace.default_individual()

# # Evaluate a randomly created individual
# training_data = create_data_by_shapes(x_shapes.values, y_shapes)
# evaluation_data = create_data_by_shapes(x_shapes.values, y_shapes)
# evaluator = partial(individual_evaluator, hyper_model, default_individual, training_data, evaluation_data)
# loss = evaluator(hyper_model, searchspace.create_random())

# print("Loss: ", loss)
