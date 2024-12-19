from enum import Enum


def apply_per_sequence(sequence, function):
    if isinstance(sequence, list):
        return [function(obj_i) for obj_i in sequence]

    if isinstance(sequence, tuple):
        return tuple([function(obj_i) for obj_i in sequence])

    if isinstance(sequence, set):
        return {function(obj_i) for obj_i in sequence}

    if isinstance(sequence, dict):
        return {k: function(v) for k, v in sequence.items()}

    return None


def object_to_dict(obj):
    result = apply_per_sequence(obj, object_to_dict)
    if result:
        return result

    if not hasattr(obj, "__dict__"):
        return obj

    if isinstance(obj, Enum):
        return obj.value

    data = {
        "class": obj.__class__.__name__,
        "module": obj.__module__,
        "vars": {name: object_to_dict(obj_i) for name, obj_i in vars(obj).items()},
    }

    # Usually this is overkill
    if hasattr(obj, "get_params"):
        data["params"] = obj.get_params(deep=True)

    return data


def object_from_dict(data):
    """
    Creates an object from a dictionary created by `object_to_dict`.
    """
    import importlib
    import inspect

    # If the data is not in the format of object data
    if not (isinstance(data, dict) and set(data.keys()).issuperset({"module", "class", "vars"})):
        # Try applying it as a sequence
        result = apply_per_sequence(data, object_from_dict)
        # If we get a result return
        if result:
            return result

    obj_class = getattr(importlib.import_module(data["module"]), data["class"])
    # Get the arguments expected by the __init__ of AquasuiteModel
    signature = inspect.signature(obj_class.__init__)
    # Get the arguments which are in the __init__
    arguments = [param.name for param in signature.parameters.values()]
    # Get the arguments which are in the vars and in the __init__
    found_arguments = {k: v for k, v in data["vars"].items() if k in arguments}
    # Get the attributes which are in the vars and not in the __init__
    found_attributes = {k: v for k, v in data["vars"].items() if k not in arguments}
    obj = obj_class(**found_arguments)

    # Set the found attributes
    for param_name, value in found_attributes.items():
        if isinstance(value, dict) and set(value.keys()).issuperset({"module", "class", "vars"}):
            value = object_from_dict(value)
        setattr(obj, param_name, value)

    # Set the params if it has the `set_params` function.
    if "params" in data.keys() and hasattr(obj_class, "set_params"):
        obj.set_params(**data["params"])
    return obj
