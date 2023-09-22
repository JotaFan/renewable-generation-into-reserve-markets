import sys
import pathlib
import importlib
import os

def get_functions_from_experiment(path_to_trained_models, struct):
    strutc_path =  pathlib.Path(os.path.join(pathlib.Path(path_to_trained_models).resolve(), struct))
    list_models =  strutc_path.glob(f'**/*.keras')
    list_models = [f for f in list_models]
    generator_script =  strutc_path.glob(f'**/*.py')
    generator_script = [f for f in generator_script]
    script = generator_script[0]
    spec = importlib.util.spec_from_file_location("module.name", script)
    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, os.path.dirname(script)) # Add the script's directory to the system path
    sys.modules[spec.name] = module 
    spec.loader.exec_module(module)

    get_dataset = getattr(module, "get_dataset", None)  # Get the "get_dataset" function from the module
    prepare_for_data = getattr(module, "prepare_for_data", None)  # Get the "get_dataset" function from the module
    prepare_for_model = getattr(module, "prepare_for_model", None)  # Get the "get_dataset" function from the module
    prediction_from_model = getattr(module, "prediction_from_model", None)
    merge_predictions = getattr(module, "merge_predictions", None)


    sys.path.remove(os.path.dirname(script))
    return get_dataset, prepare_for_data, prepare_for_model, prediction_from_model, merge_predictions
