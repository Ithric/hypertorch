import torch
from typing import List, Any, Dict, Optional
import numpy as np
from functools import partial
import copy
from . import searchspaceprimitives
from .constants import *
from dataclasses import dataclass


@dataclass
class MaterializationContext:
    path : List[str]
    individual : Dict[str,Any]
    original_forward : Any = None

    def get_child_context(self, child_name : str) -> 'MaterializationContext':
        child_path_str = "/".join(self.path + [child_name])
        if child_name not in self.individual:            
            raise ValueError(f"Missing individual at '{child_path_str}'")
        return MaterializationContext(self.path + [child_name], self.individual[child_name])
    
    def clone_with(self, **kwargs) -> 'MaterializationContext':
        new_dict = copy.deepcopy(self.__dict__)
        new_dict.update(kwargs)
        new_ctx = MaterializationContext(**new_dict)
        return new_ctx



class Individual(object):
    def __init__(self, individual_dict : Dict[str,Any]):
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
    def coalesce(left : 'Individual', right : 'Individual') -> 'Individual':
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
    def parse(individual_dict : Dict[str,Any]) -> 'Individual':
        key_values = dict()
        for key,value in individual_dict.items():
            if isinstance(value, dict):
                key_values[key] = Individual.parse(value)
            else:
                key_values[key] = value
        return Individual(key_values)


    def __getitem__(self, index) -> Any:
        return self.__individual[index]

    def __setitem__(self, index, value) -> None:
        self.__individual[index] = value

    def get(self, key, default_value=None) -> Any:
        return self.__individual.get(key, default_value)
    
    def as_dict(self) -> Dict[str,Any]:
        def rec_as_dict(d):
            output = []
            for key,item in d.__individual.items():
                if isinstance(item,Individual): 
                    output.append((key,rec_as_dict(item)))
                else: 
                    output.append((key,item))
            return dict(output)
                
        return rec_as_dict(self)


class NullSpace(object):
    pass

class SearchSpace(object):
    def __init__(self, type_key : str, ss_dict : Dict[str,Any] = None):
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

    def append_child(self, key : str, ss : 'SearchSpace') -> None:
        self._searchspace_dict[key] = ss
        
    def default_individual(self, default_values : Dict[str,Any] = DefaultLayerConstants) -> Individual:
        def rec_collapse_searchspace(searchspace : SearchSpace) -> Individual:            
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
                    elif isinstance(left, searchspaceprimitives.FloatSpace) and left.default is not None: key_values[key] = left.default
                    elif isinstance(left, searchspaceprimitives.IntSpace) and left.default is not None: key_values[key] = left.default
                    elif isinstance(left, searchspaceprimitives.NoSpace): key_values[key] = left.exact_value
                    else: raise Exception("Unknown left type: {}, key={}".format(left, key))

                elif right is not None:
                    key_values[key] = right

                else:
                    raise Exception("This is not possible")

            return Individual(key_values)
    
        return rec_collapse_searchspace(self)

    def create_random_individual(self) -> Individual:
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

    def get(self, key : str, default_value : Optional[Any] = None) -> Any:
        return self._searchspace_dict.get(key, default_value)


class HyperModel(torch.nn.Module):
    def __init__(self, debug_name : str = ""):
        super(HyperModel, self).__init__()
        self.output_shape = None
        self.debug_name = debug_name

    def materialize(self, individual : Individual, *args, **kwargs):
        orig_forward_map = {}

        def recursive_apply_materialization_context(path : List[str], obj : Any, ctx : Optional[MaterializationContext]):
            if not isinstance(obj, HyperModel): return
            module : HyperModel = obj

            # Apply the materialization context to this model            
            if ctx is not None and hasattr(module, "materializing_forward"):
                orig_forward_map[module] = module.forward
                module.forward = partial(module.materializing_forward, ctx.clone_with(original_forward=module.forward))
            elif ctx is None and module in orig_forward_map:
                module.forward = orig_forward_map[module]

            # Recursively apply to children
            for name, module in obj.named_children():
                child_path = path + [name]
                try:
                    if isinstance(module, HyperModel):
                        child_ctx = ctx.get_child_context(name) if ctx is not None else None
                        recursive_apply_materialization_context(child_path, module, child_ctx)

                    elif isinstance(module, torch.nn.ModuleDict):
                        dict_context = ctx.get_child_context(name) if ctx is not None else None
                        for key, value in module.items():
                            child_ctx = dict_context.get_child_context(key) if dict_context is not None else None                            
                            recursive_apply_materialization_context(child_path, value, child_ctx)

                    elif isinstance(module, torch.nn.ModuleList):
                        iter_context = ctx.get_child_context(name) if ctx is not None else None
                        for i, value in enumerate(module):
                            list_index_name = f"{i}"
                            child_ctx = iter_context.get_child_context(list_index_name) if ctx is not None else None
                            recursive_apply_materialization_context(child_path, value, child_ctx)
                except Exception as e:
                    raise Exception(f"Error while applying materialization context to {child_path}: {e}")


        try:
            recursive_apply_materialization_context([self.debug_name], self, MaterializationContext(["root"],individual.as_dict()))
            # Call forward again to materialize the model
            self.forward(*args, **kwargs)
        except Exception as e:
            raise Exception(f"Error while materializing {self.debug_name}: {e}")
        finally:
            recursive_apply_materialization_context([self.debug_name], self, None)

    def get_searchspace(self, default_space : Dict[str,Any]) -> Optional[SearchSpace]:
        """ What are you doing in here? In v2 you must use "build _searchspace" instaed to start the building process """
        return None


    def build_searchspace(self, default_layer_searchspace = DefaultLayerSpace):
        """ Return a representation of the searchspace of the model """
        
        def rec_merge_space(obj : HyperModel | Any, default_searchspace, path="") -> Optional[SearchSpace]:
            if not isinstance(obj, HyperModel): return None

            # Build the searchspace for this model
            searchspace = obj.get_searchspace(default_layer_searchspace)
            searchspace = searchspace or SearchSpace(obj.__class__.__name__)

            # Recursively merge children
            for name, module in obj.named_children():
                if isinstance(module, HyperModel):
                    childss = rec_merge_space(module, default_searchspace, "{}/{}".format(path,name))
                    if childss != None: 
                        searchspace.append_child(name, childss)
                
                elif isinstance(module, torch.nn.ModuleDict):
                    dict_ss = SearchSpace(name)
                    for key, value in module.items():
                        childss = rec_merge_space(value, default_searchspace, "{}/{}".format(path,name))
                        if childss != None:
                            dict_ss.append_child(key, childss)
                            
                    searchspace.append_child(name, dict_ss)

                elif isinstance(module, torch.nn.ModuleList):
                    list_ss = SearchSpace(name)
                    for i, value in enumerate(module):
                        childss = rec_merge_space(value, default_searchspace, "{}/{}".format(path,name))
                        if childss != None:
                            list_ss.append_child(str(i), childss)
                        else:
                            list_ss.append_child(str(i), NoSpace("NoSpace"))
                    searchspace.append_child(name, list_ss)

            return searchspace



        # Build the searchspace from all underlying hyper models
        return rec_merge_space(self, default_layer_searchspace)
    