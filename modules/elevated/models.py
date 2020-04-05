import torch
import numpy as np
from functools import partial
import copy


class Individual(object):
    def __init__(self, individual_dict):
        self.__individual = individual_dict

    def __str__(self):
        def rec_print(d, indent):
            output = []
            for key,item in d.__individual.items():
                if isinstance(item,Individual): 
                    inner = rec_print(item,indent + "  ")
                    output.append("{}\"{}\": {{\n{}{}\n{}}}".format(indent, key, indent, inner, indent))
                else: 
                    output.append("{}\"{}\": {}".format(indent, key, item))
            return ",\n".join(output)
                

        return "{\n" + rec_print(self, indent="  ") + "\n}"
        
    @staticmethod
    def coalesce(left, right):
        """ Coalesce two individuals together """
        all_keys = set(left.__individual.keys()).union(set(right.__individual.keys()))
        key_values = dict()
        for key in all_keys:
            left_value = left.get(key, None)
            right_value = right.get(key, None)
            coalesced_value = right_value or left_value

            if isinstance(coalesced_value, Individual):
                # Coalesce recursively if both are individuals - else, just pick the coalesced value
                if isinstance(left_value, Individual) and isinstance(right_value, Individual):
                    coalesced_value = Individual.coalesce(left_value, right_value)

            key_values[key] = coalesced_value

        return Individual(key_values)

    @staticmethod
    def parse(individual_dict):
        key_values = dict()
        for key,value in individual_dict.items():
            if isinstance(value, dict):
                key_values[key] = Individual.parse(value)
            else:
                key_values[key] = value
        return Individual(key_values)


    def __getitem__(self, index):
        return self.__individual[index]

    def __setitem__(self, index, value):
        self.__individual[index] = value

    def get(self, key, default_value=None):
        return self.__individual.get(key, default_value)


class DefaultRangeConstants(object):
    IntSpaceDefault = (1,500)
    FloatSpaceDefault = (0.0,500.0)


class SearchSpacePrimitives(object):

    class IntSpace(object):
        default_range = "default_intspace"

        def __init__(self, from_int, to_int):
            self.from_int = from_int
            self.to_int = to_int

        def __str__(self):
            return "IntSpace(from_int={}, to_int={})".format(self.from_int, self.to_int)

    class FloatSpace(object):
        default_range = "default_floatspace"

        def __init__(self, from_float, to_float):
            self.from_float = from_float
            self.to_float = to_float

        def __str__(self):
            return "FloatSpace(from_float={}, to_float={})".format(self.from_float, self.to_float)

DefaultLayerConstants = {
    "ElevatedLinear" : { "nodes" : 64 }
}

class SearchSpace(object):
    def __init__(self, type_key, key, ss_dict=None):
        self.__type_key = type_key
        self.__key = key
        self._searchspace_dict = ss_dict or dict()

    def __str__(self):
        def recPrintSearchSpace(d, indent):
            output = []
            for key,item in d._searchspace_dict.items():
                if isinstance(item,SearchSpace): 
                    inner = recPrintSearchSpace(item,indent+"  ")
                    output.append("{}\"{}#{}\": {{\n{}{}\n{}}}".format(indent, item.__type_key, key, indent, inner, indent))
                else: 
                    output.append("{}\"{}\": {}".format(indent, key, item))
            return "\n".join(output)

        return "\"{}#{}\"".format(self.__type_key, self.__key) + " : {\n" + recPrintSearchSpace(self, indent="  ") + "\n}"

    def append_child(self, key, ss):
        self._searchspace_dict[key] = ss

    def default_individual(self, default_values = DefaultLayerConstants) -> Individual:
        def rec_collapse_searchspace(searchspace) -> Individual:
            if searchspace.__type_key in default_values:
                return Individual.parse(default_values[searchspace.__type_key])
            else:
                key_values = {}
                for key,value in searchspace._searchspace_dict.items():
                    if isinstance(value, SearchSpace):
                        key_values[key] = rec_collapse_searchspace(value)
                    else:
                        raise Exception("Unknown type:", value)

                return Individual(key_values)
    
        return rec_collapse_searchspace(self)

    def create_random(self, default_searchspace = DefaultRangeConstants()) -> Individual:
        def rec_collapse_searchspace_to_random(searchspace : SearchSpace) -> Individual:
            key_values = dict()
            for key,value in searchspace._searchspace_dict.items():
                if isinstance(value, SearchSpace):
                    key_values[key] = rec_collapse_searchspace_to_random(value)
                elif isinstance(value, SearchSpacePrimitives.IntSpace):
                    key_values[key] = np.random.randint(value.from_int, value.to_int)
                else:
                    raise Exception("Unknown type:", type(value))


            return Individual(key_values)

        return rec_collapse_searchspace_to_random(self)

        
class MaterializedModel(torch.nn.Module):
    def __init__(self, elevated_model, torch_modules):
        super(MaterializedModel, self).__init__()
        self.elevated_model = elevated_model
        for idx,(layer_key,module) in enumerate(torch_modules):
            self.add_module("{}#{}".format(idx,layer_key), module)

    def forward(self, x):
        return self.elevated_model(x)

    


class ElevatedModel(object):
    def __init__(self, name):
        super(ElevatedModel, self).__init__()
        self.__name = name
        pass

    @property
    def Name(self):
        return self.__name

    def forward(self, x):
        pass

    def __get_elevated_models(self):
        return [var for var in vars(self).items() if isinstance(var[1], ElevatedModel)]

    def __get_torch_modules(self):
        return [var for var in vars(self).items() if isinstance(var[1], torch.nn.Module)]

    def __call__(self, x):
        return self.forward(x)

    def materialize(self, individual : Individual, input_shapes, torch_module_list=[]) -> MaterializedModel:
        """ Turn this higher order model into a regular torch.Module """
        if not hasattr(self, "original_forward"): self.original_forward = self.forward
        material_self = copy.copy(self)
        # material_self = self # TODO: Make this function pure

        def data_to_shape(data):
            if isinstance(data, (tuple,list)):
                return [d.shape[1:] for d in data]
            else:
                return data.shape[1:]

        def shapes_to_sampledata(shapes):
            if isinstance(input_shapes, (tuple,list)) and not isinstance(input_shapes, torch.Size):
                return [torch.from_numpy(np.zeros((1,)+shape, dtype=np.float32)) for shape in input_shapes]
            else:
                return torch.from_numpy(np.zeros((1,)+input_shapes, dtype=np.float32))
                

        def materializing_forward(variable_name, elevated_model_instance, original_forward, data):
            if elevated_model_instance.ismaterializing == True:
                tmp = elevated_model_instance.materialize(individual.get(elevated_model_instance.Name,None), data_to_shape(data))
                if isinstance(tmp, MaterializedModel):
                    torch_module_list.extend(tmp.elevated_model.__get_torch_modules())
                else:
                    torch_module_list.extend(tmp.__get_torch_modules())

                return original_forward(data)
            else:
                return original_forward(data)

        for variable_name,elevated_model in material_self.__get_elevated_models():
            if not hasattr(elevated_model, "original_forward"): 
                elevated_model.original_forward = elevated_model.forward
                elevated_model.forward = partial(materializing_forward, variable_name, elevated_model, elevated_model.forward)

            elevated_model.ismaterializing = True

        # Perform a materializing forward
        fake_data = shapes_to_sampledata(input_shapes)
        material_self.original_forward(fake_data)

        # Disable materialization
        for _,elevated_model in material_self.__get_elevated_models():
            elevated_model.ismaterializing = False

        return MaterializedModel(material_self, torch_module_list)

    def get_searchspace(self, default_searchspace_constants = DefaultRangeConstants()):
        """ Return a representation of the searchspace of the model """
        def apply_defaults(searchspace, default_searchspace):
            for key,value in searchspace._searchspace_dict.items():
                if isinstance(value, SearchSpace):
                    value = apply_defaults(value, default_searchspace)
                else:
                    if value == SearchSpacePrimitives.IntSpace.default_range: value = SearchSpacePrimitives.IntSpace(*default_searchspace_constants.IntSpaceDefault)
                    elif value == SearchSpacePrimitives.FloatSpace.default_range: value = SearchSpacePrimitives.FloatSpace(*default_searchspace_constants.FloatSpaceDefault)

            searchspace._searchspace_dict[key] = value
            return searchspace


        ss = SearchSpace(self.__class__.__name__, self.__name)

        elevated_models = [var for var in vars(self).items() if isinstance(var[1], ElevatedModel)]
        for key,value in elevated_models:
            childss = value.get_searchspace()
            if childss != None: 
                ss.append_child(value.Name, childss)

        ss = apply_defaults(ss, default_searchspace_constants)
        return ss

