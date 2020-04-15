import torch
import numpy as np
from functools import partial
import copy
from hypertorch import searchspaceprimitives
from hypertorch.constants import *


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



class SearchSpace(object):
    def __init__(self, type_key, key, ss_dict=None):
        self.__type_key = type_key
        self._key = key
        self._searchspace_dict = ss_dict or dict()
        self.type_key = type_key

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

        return "\"{}#{}\"".format(self.__type_key, self._key) + " : {\n" + recPrintSearchSpace(self, indent="  ") + "\n}"

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

    def create_random(self) -> Individual:
        def rec_collapse_searchspace_to_random(searchspace : SearchSpace) -> Individual:
            key_values = dict()
            for key,value in searchspace._searchspace_dict.items():
                if isinstance(value, SearchSpace):
                    key_values[key] = rec_collapse_searchspace_to_random(value)
                elif isinstance(value, searchspaceprimitives.IntSpace):
                    key_values[key] = np.random.randint(value.from_int, value.to_int)
                elif isinstance(value, searchspaceprimitives.FloatSpace):
                    key_values[key] = np.random.uniform(value.from_float, value.to_float)
                elif isinstance(value, searchspaceprimitives.NoSpace):
                    key_values[key] = value.exact_value
                else:
                    raise Exception("Unknown type:", type(value))

            return Individual(key_values)

        return rec_collapse_searchspace_to_random(self)

    def get(self, key, default_value=None):
        return self._searchspace_dict.get(key, default_value)

        
class MaterializedModel(torch.nn.Module):
    def __init__(self, hyper_model, torch_modules):
        super(MaterializedModel, self).__init__()
        self.hyper_model = hyper_model
        for idx,(layer_key,module) in enumerate(torch_modules):
            self.add_module("{}#{}".format(idx,layer_key), module)

    def forward(self, x):
        return self.hyper_model(x)
        
        
    def train(self, mode=True):
        self.hyper_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.hyper_model.train(mode=False)
        return super().eval()

    


class HyperModel(object):
    def __init__(self, name):
        super(HyperModel, self).__init__()
        self.__name = name
        self.training = True
        pass
    
    def __get_hyper_models(self):
        return [var for var in vars(self).items() if isinstance(var[1], HyperModel)]

    def __get_torch_modules(self):
        return [var for var in vars(self).items() if isinstance(var[1], torch.nn.Module)]

    @property
    def Name(self):
        return self.__name
    
    def train(self, mode):
        self.training = mode
        for _,hyper_child in self.__get_hyper_models():
            hyper_child.train(mode)

    def forward(self, x):
        pass


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
                

        def materializing_forward(variable_name, hyper_model_instance, original_forward, data):
            if hyper_model_instance.ismaterializing == True:
                tmp = hyper_model_instance.materialize(individual.get(hyper_model_instance.Name,None), data_to_shape(data))
                if isinstance(tmp, MaterializedModel):
                    torch_module_list.extend(tmp.hyper_model.__get_torch_modules())
                else:
                    torch_module_list.extend(tmp.__get_torch_modules())

                return original_forward(data)
            else:
                return original_forward(data)

        for variable_name,hyper_model in material_self.__get_hyper_models():
            if not hasattr(hyper_model, "original_forward"): 
                hyper_model.original_forward = hyper_model.forward
                hyper_model.forward = partial(materializing_forward, variable_name, hyper_model, hyper_model.forward)

            hyper_model.ismaterializing = True

        # Perform a materializing forward
        fake_data = shapes_to_sampledata(input_shapes)
        material_self.original_forward(fake_data)

        # Disable materialization
        for _,hyper_model in material_self.__get_hyper_models():
            hyper_model.ismaterializing = False

        return MaterializedModel(material_self, torch_module_list)

    def get_searchspace(self, default_layer_searchspace = DefaultLayerSpace):
        """ Return a representation of the searchspace of the model """

        def rec_merge_space(searchspace, default_searchspace, path=""):
            default_value = default_searchspace.get(searchspace.type_key, searchspace)
            for key,value in searchspace._searchspace_dict.items():
                if isinstance(value, SearchSpace):
                    value = rec_merge_space(value, default_searchspace, "{}/{}".format(path,searchspace._key))
                elif value is None: value = default_value.get(key,None)
                if value is None: raise Exception("Undefined searchspace on {}/{}#{}".format(path, searchspace.type_key, searchspace._key))
                searchspace._searchspace_dict[key] = value
            return searchspace


        # Build the searchspace from all underlying hyper models
        ss = SearchSpace(self.__class__.__name__, self.__name)
        hyper_models = [var for var in vars(self).items() if isinstance(var[1], HyperModel)]
        for key,value in hyper_models:
            childss = value.get_searchspace()
            if childss != None: 
                ss.append_child(value.Name, childss)

        # Merge with default searchspace per layer (where value is Null -> failover to default space)
        ss = rec_merge_space(ss, default_layer_searchspace, "..")
        return ss

