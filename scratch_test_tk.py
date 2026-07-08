import threading
import tkinter as tk
from tkinter import filedialog

def thread_func():
    print("Opening dialog from thread...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.asksaveasfilename(defaultextension=".tlp")
    root.destroy()
    print("Path:", path)

t = threading.Thread(target=thread_func)
t.start()
t.join()
