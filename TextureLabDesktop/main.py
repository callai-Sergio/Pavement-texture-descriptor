"""
TextureLab Desktop – Entry Point

Standalone pavement texture analysis application wrapping the Streamlit UI.
"""
import sys
import os
import threading
import subprocess
import time
import socket

# Add parent directory to path so engine/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import webview
except ImportError:
    print("Error: pywebview is not installed. Please run 'pip install pywebview'")
    sys.exit(1)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_streamlit(port, stop_event):
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # We use subprocess to run Streamlit
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TextureLab", "app.py")
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path], env=env)
    
    # Wait until the main thread tells us to stop
    stop_event.wait()
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def main():
    # Multiprocessing support for Windows/PyInstaller
    import multiprocessing
    multiprocessing.freeze_support()

    # Find a free port
    port = find_free_port()
    
    # Stop event to kill the subprocess gracefully
    stop_event = threading.Event()
    
    # Start Streamlit in the background
    t = threading.Thread(target=start_streamlit, args=(port, stop_event))
    t.daemon = True
    t.start()
    
    # Give the server a couple seconds to start up
    time.sleep(2)
    
    url = f"http://localhost:{port}"
    
    # Enable downloads for Streamlit download buttons
    webview.settings['ALLOW_DOWNLOADS'] = True
    
    # Create the webview window
    webview.create_window(
        title="TextureLab Desktop",
        url=url,
        width=1280,
        height=800,
        min_size=(800, 600),
        background_color='#0f0f1a'
    )
    
    # Start the webview loop (blocks until the window is closed)
    webview.start(private_mode=False)
    
    # Shutdown the streamlit server
    stop_event.set()
    t.join(timeout=3)


if __name__ == "__main__":
    main()
