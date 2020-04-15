
from hypertorch.searchspaceprimitives import *

# Used to map layer types to a default searchspace for the type
DefaultLayerSpace = {
    "HyperLinear" : { "nodes" : IntSpace(1,200) },
    "HyperDropout" : { "dropout_p" : FloatSpace(0,1) },
    "GaussianNoise" : { "sigma" : FloatSpace(0,1) }
}

# Used to map a searchspace to a "default individual" (SearchSpace.default_individual, default_values=this)
DefaultLayerConstants = {
    "HyperLinear" : { "nodes" : 64 },
    "HyperDropout" : { "dropout_p" : 0 },
    "GaussianNoise" : { "sigma" : 0 }
}