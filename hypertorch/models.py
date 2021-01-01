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
                    output.append("{}\"{}\": {{\n{}\n{}}}".format(indent, key, inner, indent))
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


class NullSpace(object):
    pass

class SearchSpace(object):
    def __init__(self, type_key, ss_dict=None):
        self.__type_key = type_key
        self._searchspace_dict = ss_dict or dict()
        self.type_key = type_key

    def __str__(self):
        def recPrintSearchSpace(d, indent):
            output = []
            for key,item in d._searchspace_dict.items():
                if isinstance(item,SearchSpace): 
                    inner = recPrintSearchSpace(item,indent + "  ")
                    output.append("{}\"{}#{}\": {{\n{}\n{}}}".format(indent, item.__type_key, key, inner, indent))
                else: 
                    output.append("{}\"{}\": {}".format(indent, key, item))
            return ",\n".join(output)

        return "\"{}#{}\"".format(self.__type_key, "root") + " : {\n" + recPrintSearchSpace(self, indent="  ") + "\n}"

    def append_child(self, key, ss):
        self._searchspace_dict[key] = ss
        
    def default_individual(self, default_values = DefaultLayerConstants) -> Individual:
        def rec_collapse_searchspace(searchspace) -> Individual:            
            key_values = {}
            default_key_values = default_values.get(searchspace.__type_key, {})
            all_keys = set(default_key_values.keys()).union(searchspace._searchspace_dict.keys())
            
            for key in all_keys:
                left = searchspace._searchspace_dict.get(key, None)
                right = default_key_values.get(key, None)

                if left is not None and right is not None: # Merge them
                    if isinstance(left, searchspaceprimitives.NoSpace): key_values[key] = left.exact_value
                    else: key_values[key] = right

                elif left is not None:
                    if isinstance(left, SearchSpace): key_values[key] = rec_collapse_searchspace(left)
                    elif isinstance(left, FloatSpace) and left.default is not None: key_values[key] = left.default
                    else: raise Exception("Unknown left type: {}, key={}".format(left, key))

                elif right is not None:
                    key_values[key] = right

                else:
                    raise Exception("This is not possible")

            return Individual(key_values)
    
        return rec_collapse_searchspace(self)

    def create_random(self) -> Individual:
        def rec_collapse_searchspace_to_random(searchspace : SearchSpace) -> Individual:
            key_values = dict()
            for key,value in searchspace._searchspace_dict.items():
                if isinstance(value, NullSpace):
                    continue
                elif isinstance(value, SearchSpace):
                    key_values[key] = rec_collapse_searchspace_to_random(value)
                elif isinstance(value, searchspaceprimitives.IntSpace):
                    key_values[key] = np.random.randint(value.from_int, value.to_int)
                elif isinstance(value, searchspaceprimitives.FloatSpace):
                    key_values[key] = np.random.uniform(value.from_float, value.to_float)
                elif isinstance(value, searchspaceprimitives.NoSpace):
                    key_values[key] = value.exact_value
                elif isinstance(value, searchspaceprimitives.OneOfSet):
                    key_values[key] = np.random.choice(value.set_values)
                else:
                    raise Exception("Unknown type under {}: {}".format(key,type(value)))

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
    materializing_hack = False

    def __init__(self, debug_name=None):
        super(HyperModel, self).__init__()
        self.training = True
        self.modules = []
        self.debug_name = debug_name or str(type(self))
        pass
    
    def __get_hyper_models(self):
        return [var for var in list(vars(self).items()) + self.modules if isinstance(var[1], HyperModel)]

    def __get_torch_modules(self):
        return [var for var in list(vars(self).items()) + self.modules if isinstance(var[1], torch.nn.Module)]
   
    
    def train(self, mode):
        self.training = mode
        for _,hyper_child in self.__get_hyper_models():
            hyper_child.train(mode)

    def forward(self, x):
        pass

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)

    def add_module(self, key, module):
        self.modules.append((key,module))
        return module

    def __recursive_apply_variable_name(self):
        for (var_name,hypermodel) in self.__get_hyper_models():
            hypermodel.__name = var_name
            hypermodel.__recursive_apply_variable_name()

    def materialize(self, individual : Individual, input_shapes, torch_module_list=None, forward_ext_args=None) -> MaterializedModel:
        """ Turn this higher order model into a regular torch.Module """
        torch_module_list = torch_module_list or []
        is_root = HyperModel.materializing_hack == False
        if is_root:
            material_self = copy.deepcopy(self)
            HyperModel.materializing_hack = True
            self.__recursive_apply_variable_name()
        else:
            material_self = self

        # Preserve the original forward function
        if not hasattr(material_self, "original_forward"): 
            material_self.original_forward = material_self.forward

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
                

        def materializing_forward(state, individual, variable_name, hyper_model_instance, original_forward, data, **kwargs):
            if variable_name in state: return original_forward(data, **kwargs)
            else: state[variable_name] = "bound"

            if hyper_model_instance.ismaterializing == True:
                indvar = individual.get(variable_name,None)
                if indvar == None: raise Exception("Individual does not contain key: {}".format(variable_name))
                tmp = hyper_model_instance.materialize(indvar, data_to_shape(data), forward_ext_args=kwargs)
                tmp.eval()
                if isinstance(tmp, torch.nn.Module):
                    torch_module_list.append((variable_name,tmp))
                else:
                    raise Exception("Materialization of {}:{} did not return a torch.nn.Module".format(variable_name, hyper_model_instance))

                return original_forward(data, **kwargs)
            else:
                return original_forward(data, **kwargs)

        binding_state = {}
        for variable_name,hyper_model in material_self.__get_hyper_models():
            if not hasattr(hyper_model, "original_forward"): 
                hyper_model.original_forward = hyper_model.forward
                hyper_model.forward = partial(materializing_forward, binding_state, individual, variable_name, hyper_model, hyper_model.forward)

            hyper_model.ismaterializing = True

        # Perform a materializing forward
        fake_data = shapes_to_sampledata(input_shapes)
        material_self.original_forward(fake_data, **(forward_ext_args or {}))

        # Add torch modules too!
        for torch_module in material_self.__get_torch_modules():
            torch_module_list.append(torch_module)

        # Disable materialization
        for _,hyper_model in material_self.__get_hyper_models():
            hyper_model.ismaterializing = False

        if is_root == True: HyperModel.materializing_hack = False
        return MaterializedModel(material_self, torch_module_list)

    def get_searchspace(self, default_layer_searchspace = DefaultLayerSpace):
        """ Return a representation of the searchspace of the model """
        self.__recursive_apply_variable_name()

        def rec_merge_space(searchspace, default_searchspace, path=""):
            default_value = default_searchspace.get(searchspace.type_key, searchspace)
            for key,value in searchspace._searchspace_dict.items():
                if isinstance(value, SearchSpace):
                    value = rec_merge_space(value, default_searchspace, "{}/{}".format(path,key))
                elif value is None: value = default_value.get(key,None)
                if value is None: raise Exception("Undefined searchspace on {}/{}#{}".format(path, searchspace.type_key, key))
                searchspace._searchspace_dict[key] = value
            return searchspace


        # Build the searchspace from all underlying hyper models
        ss = SearchSpace(self.__class__.__name__)
        hyper_models = self.__get_hyper_models() # [var for var in vars(self).items() if isinstance(var[1], HyperModel)]
        for key,value in hyper_models:
            childss = value.get_searchspace()
            if childss != None: 
                ss.append_child(key, childss)

        # Merge with default searchspace per layer (where value is Null -> failover to default space)
        ss = rec_merge_space(ss, default_layer_searchspace, "..")
        return ss

