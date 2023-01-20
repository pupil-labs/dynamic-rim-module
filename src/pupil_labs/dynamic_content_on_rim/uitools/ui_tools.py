import logging
import os
import pathlib
import platform
import tkinter as tk
import uuid
from tkinter import filedialog
from typing import Optional

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table


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


def rich_df(
    pandas_dataframe: pd.DataFrame,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Based on https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938"""
    rich_table = Table(show_header=True, header_style="bold magenta")

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    rich_table.row_styles = ["none", "dim"]
    rich_table.box = box.SIMPLE_HEAD
    console = Console(width=150)
    console.print(rich_table)
    return rich_table
