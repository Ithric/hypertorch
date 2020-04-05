# Hypertorch 
Hypertorch is created to make building and optimizing pytorch modules easier by "lifting" the model itself such that the shape of each layer is determined at runtime, rather than by the programmer. This enables automatic searchspace inference, and simplifies the integration of black box optimization frameworks

## Basic usage
Start by defining the model:

```python
class MyTestModel(elevated.ElevatedModel):
    def __init__(self):
        super(MyTestModel, self).__init__("root")
        self.layer_a = ElevatedLinear("layer_a")
        self.layer_b = ElevatedLinear("layer_b")
        self.layer_c = ElevatedLinear("layer_c", n_output_nodes=1)
        pass

    def forward(self, x):
        y = self.layer_a(x[0])
        y = self.layer_b(y)
        y = self.layer_c(y)
        return y
```

The elevated model needs to be materialized before it can be used:

```python
# Create the hypermodel
elevated_model = MyTestModel()
searchspace = elevated_model.get_searchspace()
individual = searchspace.default_individual()

# Materialize the model - returns a regular pytorch nn.Module
model = elevated_model.materialize(individual, [kx.shape[1:] for kx in inputs])
```