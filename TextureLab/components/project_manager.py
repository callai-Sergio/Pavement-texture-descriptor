"""
project_manager.py - Save and Load Workspace projects for TextureLab
"""
import pickle
import io

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
