import logging
import os
import pathlib
import platform
import tkinter as tk
import uuid
from tkinter import filedialog


def get_path(msg, file, _path):
    root = tk.Tk()
    root.withdraw()
    arguments = {"title": msg}
    if platform.system() == "Darwin":
        arguments["message"] = msg
    if _path is None:
        _path = filedialog.askdirectory(**arguments)
    if not _path:
        warning = "User aborted directory selection"
        logging.warning(warning)
        raise SystemExit(warning)
    if not os.path.exists(os.path.join(_path, file)):
        error = f"Could not find file {file} in selected folder"
        logging.error(error)
        raise SystemExit(error)
    return _path


def get_file(path):
    if path is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select the screen video",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi")],
        )
    return path


def get_savedir(path, type):
    if path is None:
        if type == "video":
            arguments = {
                "defaultextension": ".mp4",
                "initialfile": (str(uuid.uuid4()) + ".mp4"),
                "title": "Select where to save the output video:",
            }
        elif type == "csv":
            arguments = {
                "defaultextension": ".csv",
                "initialfile": ("gaze.csv"),
                "title": "Select where to save the output csv file:",
            }
        arguments["initialdir"] = pathlib.Path.home()
        path = filedialog.asksaveasfilename(**arguments)
    return path


def progress_bar(current, total, label="", bar_length=20):
    """
    Prints a progress bar.
    :param current: The current progress.
    :param total: The total progress.
    :param bar_length: The length of the progress bar in the cmd.
    """
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * "-" + "✈︎"
    padding = int(bar_length - len(arrow)) * " "
    ending = "\n" if current == total else "\r"
    print(f"Progress {label}: [{arrow}{padding}] {int(fraction*100)}%", end=ending)
