"""
project_manager.py - Save and Load Workspace projects for TextureLab
"""
import pickle
import io

def _fix_streamlit_class_reload(obj):
    """
    Recursively updates the __class__ reference of custom objects to match the
    currently loaded module in sys.modules. This prevents PicklingError when 
    Streamlit reloads a module and creates a class identity mismatch.
    """
    import sys
    if isinstance(obj, list):
        for item in obj:
            _fix_streamlit_class_reload(item)
    elif isinstance(obj, dict):
        for val in obj.values():
            _fix_streamlit_class_reload(val)
    elif hasattr(obj, "__class__"):
        cls = obj.__class__
        mod = cls.__module__
        if mod and (mod.startswith("src.") or mod.startswith("engine.")) and mod in sys.modules:
            curr_mod = sys.modules[mod]
            if hasattr(curr_mod, cls.__name__):
                obj.__class__ = getattr(curr_mod, cls.__name__)

def export_project(session_state_dict: dict) -> bytes:
    """
    Serializes the necessary parts of the session state to a bytes object
    so the user can download it as a .tlp (TextureLab Project) file.
    """
    keys_to_save = [
        "surfaces",
        "file_names",
        "profiles",
        "results_1d",
        "results_areal",
        "aggregated",
        "batch_agg",
        "warnings",
        "processed",
        "logs",
        "selected_params",
        "chart_styles",
    ]
    
    project_data = {k: session_state_dict.get(k) for k in keys_to_save if k in session_state_dict}
    
    # Fix class references before pickling
    _fix_streamlit_class_reload(project_data)
    
    buffer = io.BytesIO()
    pickle.dump(project_data, buffer)
    return buffer.getvalue()

def load_project(file_bytes: bytes, session_state_ref) -> None:
    """
    Loads a .tlp file and restores the session state.
    """
    buffer = io.BytesIO(file_bytes)
    project_data = pickle.load(buffer)
    
    for k, v in project_data.items():
        session_state_ref[k] = v
