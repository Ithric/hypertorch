import torch
from typing import List, Any, Dict, Optional
import numpy as np
from functools import partial
import copy
from . import searchspaceprimitives
from .searchspaceprimitives import SearchSpaceKind
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
    def __init__(self, type_key : str, ss_dict : Dict[str,Any] = None, labels : Optional[str] = None):
        self.__type_key = type_key
        self._searchspace_dict = ss_dict or dict()
        self.__labels = labels or []
        self.type_key = type_key

    @property
    def labels(self) -> List[str]:
        return self.__labels
    
    def add_labels(self, *labels : List[str]) -> None:
        self.__labels.extend(labels)

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
                    else: raise Exception("Unknown left type: {} {}, key={}".format(left, type(left), key))

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

    def get(self, key : str, default_value : Optional[Any] = None) -> 'SearchSpace':
        return self._searchspace_dict.get(key, default_value)
    
    def select_by_label(self, target : str, kind_or_label : List[SearchSpaceKind|str], mode : str) -> Optional['SearchSpace']: 
        """ Selects a subset of the searchspace based on the kind or label of the primitives
        
        Args:
            target (str): Either "hypermodules" or "primitives"
            kind_or_label (List[SearchSpaceKind|str]): The kind or label to select by
            mode (str): Either "include" or "exclude"
        """
        assert target in ["hypermodules", "primitives"], "Target must be either 'searchspace' or 'primitives'"
        assert mode in ["include", "exclude"], "Mode must be either 'include' or 'exclude'"
        def label_is_valid(labels : List[str]):
            if mode == "include":
                return len(set(kind_or_label).intersection(set(labels))) > 0
            elif mode == "exclude":
                return len(set(kind_or_label).intersection(set(labels))) == 0
            
        if target == "hypermodules" and not label_is_valid(self.labels):
            return None

        output_dict = {}
        for key, value in self._searchspace_dict.items():
            if isinstance(value, SearchSpace):                
                sschild = value.select_by_label(target, kind_or_label, mode=mode)
                if sschild is not None: output_dict[key] = sschild
                
            elif isinstance(value, searchspaceprimitives.SearchSpacePrimitive):
                if target != "primitives" or label_is_valid(value.kind):
                    output_dict[key] = value

        if len(output_dict) == 0: return None
        return SearchSpace(self.__type_key, output_dict, labels=self.labels)
        
    # override compare equals
    def deep_equals(self, other : 'SearchSpace'):
        import json
        assert isinstance(other, SearchSpace), "Other must be a SearchSpace"
        return json.dumps(str(self)) == json.dumps(str(other)) # Hacky, but it works for now


def resolve_path(current_path: List[str], target_path: str) -> List[str]:
    """Resolve a target path to an absolute path."""
    if target_path.startswith("/"):
        # Absolute path: Return it as is, split into components, but filter out empty strings
        return [component for component in target_path.split("/") if component]
    else:
        # Relative path: Resolve it relative to the current path
        resolved_path = list(current_path)  # Create a mutable copy of current_path
        for component in target_path.split("/"):
            if component == "..":
                # Move up one directory (remove the last component), if possible
                if resolved_path:
                    resolved_path.pop()
            elif component != ".":
                # Ignore "." since it represents the current directory
                resolved_path.append(component)
        return resolved_path


class HyperModel(torch.nn.Module):
    def __init__(self, debug_name : str = "", labels : Optional[List[str]] = None, rebase_searchspace_root : Optional[str] = None):
        super(HyperModel, self).__init__()
        self.output_shape = None
        self.debug_name = debug_name
        self.__labels = labels or []
        self.__rebase_searchspace_root = rebase_searchspace_root        

    @property
    def labels(self) -> List[str]:
        return self.__labels
    
    def set_rebase_root(self, path : str) -> None:
        self.__rebase_searchspace_root = path

    def materialize(self, individual : Individual, *args, **kwargs):
        orig_forward_map = {}
        validation_counter = 0
        materialized_set = set()
        root_ctx = MaterializationContext([],individual.as_dict())
        
        def recursive_reset_forward(obj : Any):
            nonlocal validation_counter
            nonlocal orig_forward_map    
            if not isinstance(obj, HyperModel): return
            
            module : HyperModel = obj
            if module in orig_forward_map:
                module.forward = orig_forward_map[module]
                validation_counter -= 1
                orig_forward_map.pop(module)

            for name, module in obj.named_children():
                if isinstance(module, HyperModel):
                    recursive_reset_forward(module)
                elif isinstance(module, torch.nn.ModuleDict):
                    for key, value in module.items():
                        recursive_reset_forward(value)
                elif isinstance(module, torch.nn.ModuleList):
                    for value in module:
                        recursive_reset_forward(value)

        def recursive_apply_materialization_context(obj : Any, ctx : MaterializationContext):
            nonlocal validation_counter
            nonlocal orig_forward_map
            nonlocal materialized_set
            if not isinstance(obj, HyperModel): return
            if obj in materialized_set: return # Already materialized
            materialized_set.add(obj)
            module : HyperModel = obj
            

            # Apply the materialization context to this model            
            if hasattr(module, "materializing_forward"):
                orig_forward_map[module] = module.forward
                
                if "materializing_forward" in str(module.forward):
                    raise Exception(f"Module {module.debug_name} has already been materialized. This is a bug.")
                
                forward_ctx = ctx.clone_with(original_forward=module.forward)
                module.forward = partial(module.materializing_forward, forward_ctx)
                validation_counter += 1

            # Recursively apply to children
            for name, child_module in obj.named_children():
                child_module : HyperModel = child_module
                child_path = ctx.path + [name]
                    
                try:
                    if isinstance(child_module, HyperModel):
                        # Rebase the searchspace if applicable
                        if child_module.__rebase_searchspace_root is not None:
                            modified_root = resolve_path(child_path, child_module.__rebase_searchspace_root)
                            child_ctx = root_ctx
                            for part in modified_root:
                                child_ctx = child_ctx.get_child_context(part)
                        else:
                            child_ctx = ctx.get_child_context(name)
                            
                        recursive_apply_materialization_context(child_module, child_ctx)

                    elif isinstance(child_module, torch.nn.ModuleDict):
                        dict_context = ctx.get_child_context(name)
                        for grandchild_name, grandchild in child_module.items():
                            # Rebase the searchspace if applicable
                            if grandchild.__rebase_searchspace_root is not None:
                                modified_root = resolve_path(ctx.path + [grandchild_name], grandchild.__rebase_searchspace_root)
                                child_ctx = root_ctx
                                for part in modified_root:
                                    child_ctx = child_ctx.get_child_context(part)
                            else:
                                child_ctx = dict_context.get_child_context(grandchild_name)
                                
                            # child_ctx = dict_context.get_child_context(key) if dict_context is not None else None                            
                            recursive_apply_materialization_context(grandchild, child_ctx)

                    elif isinstance(child_module, torch.nn.ModuleList):
                        iter_context = ctx.get_child_context(name)
                        for i, value in enumerate(child_module):
                            list_index_name = f"{i}"
                            child_ctx = iter_context.get_child_context(list_index_name)
                            recursive_apply_materialization_context(value, child_ctx)
                except Exception as e:
                    raise Exception(f"Error while applying materialization context to {child_path}: {e}")


        try:
            recursive_apply_materialization_context(self, root_ctx)
            # Call forward again to materialize the model
            self.forward(*args, **kwargs)
        except Exception as e:
            raise Exception(f"Error while materializing {self.debug_name}: {e}")
        finally:
            recursive_reset_forward(self)
            assert validation_counter == 0, f"Validation counter should be 0, but its {validation_counter} - This is a bug"
            

    def get_searchspace(self, default_space : Dict[str,Any]) -> Optional[SearchSpace]:
        """ What are you doing in here? In v2 you must use "build _searchspace" instaed to start the building process """
        return None


    def build_searchspace(self, default_layer_searchspace = DefaultLayerSpace):
        """ Return a representation of the searchspace of the model """
        
        rebased_searchspaces : Dict[str,SearchSpace] = {}
        
        def rec_merge_space(obj : HyperModel | Any, default_searchspace, path : List[str]) -> Optional[SearchSpace]:
            if not isinstance(obj, HyperModel): return None

            # Build the searchspace for this model
            searchspace = obj.get_searchspace(default_layer_searchspace)
            searchspace = searchspace or SearchSpace(obj.__class__.__name__)

            # Apply labels from the hypermodel (if applicable) to the searchspace
            if isinstance(obj, HyperModel):
                searchspace.add_labels(*obj.labels)
            
            # Recursively merge children
            for name, module in obj.named_children():
                if isinstance(module, HyperModel):
                    childss = rec_merge_space(module, default_searchspace, path + [name])
                    if childss != None: 
                        searchspace.append_child(name, childss)
                
                elif isinstance(module, torch.nn.ModuleDict):
                    dict_ss = SearchSpace("ModuleDict")
                    for key, value in module.items():
                        childss = rec_merge_space(value, default_searchspace, path + [name])
                        if childss != None:
                            dict_ss.append_child(key, childss)
                            
                    searchspace.append_child(name, dict_ss)

                elif isinstance(module, torch.nn.ModuleList):
                    list_ss = SearchSpace("ModuleList")
                    for i, value in enumerate(module):
                        childss = rec_merge_space(value, default_searchspace, path + [name])
                        if childss != None:
                            list_ss.append_child(str(i), childss)
                        else:
                            list_ss.append_child(str(i), NoSpace("NoSpace"))
                    searchspace.append_child(name, list_ss)
                    
            if isinstance(obj, HyperModel) and obj.__rebase_searchspace_root is not None:
                target_path = "/".join(resolve_path(path, obj.__rebase_searchspace_root))
                if target_path in rebased_searchspaces and  rebased_searchspaces[target_path].deep_equals(searchspace) == False:
                    raise Exception(f"Rebase path already exists: {target_path}, but the searchspaces are different: {rebased_searchspaces[target_path]} != {searchspace}")
                # print(f"Rebasing {obj.debug_name} to {target_path}")
                rebased_searchspaces[target_path] = searchspace
                return None # Do not include this in the searchspace

            return searchspace



        # Build the searchspace from all underlying hyper models
        final_searchspace = rec_merge_space(self, default_layer_searchspace, [])
        
        # Rebase the searchspaces
        for rebased_key, rebased_ss in rebased_searchspaces.items():
            rebased_path = [a for a in rebased_key.split("/") if a != ""]
            current_path = final_searchspace
            for part_idx, part_name in enumerate(rebased_path):
                if part_idx == len(rebased_path) - 1:
                    current_path.append_child(part_name, rebased_ss)
                    break
                
                # keep going down the path
                current_path = current_path.get(part_name)
                if current_path is None: 
                    raise Exception(f"Rebase path not found: {rebased_key} - {rebased_path[:part_idx]}")
            
        
        return final_searchspace
    