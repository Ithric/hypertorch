
from .searchspaceprimitives import *

# Used to map layer types to a default searchspace for the type
DefaultLayerSpace = {
    "HyperLinear" : { "nodes" : IntSpace(1,200) },
    "HyperDropout" : { "dropout_p" : FloatSpace(0,0.999) },
    "HyperGaussNoise" : { "sigma" : FloatSpace(0,2) },
    "HyperRNN" : {
        "mode" : OneOfSet(["LSTM", "GRU"]),
        "ndims" : IntSpace(1, 3),
        "bidir" : OneOfSet([True, False]),
        "nlayers" : IntSpace(1, 3)
    },
    "HyperNodeSelector" : {},
    "HyperNoOp" : {}
}


# Used to map a searchspace to a "default individual" (SearchSpace.default_individual, default_values=this)
DefaultLayerConstants = {
    "HyperLinear" : { "nodes" : 64 },
    "HyperDropout" : { "dropout_p" : 0.1 },
    "HyperGaussNoise" : { "sigma" : 0.1 },
    "HyperNodeSelector" : { "key" : "default_key" },
    "HyperRNN" : {
        "mode" : "GRU",
        "ndims" : 64,
        "bidir" : False,
        "nlayers" : 1
    }
}