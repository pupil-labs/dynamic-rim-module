import argparse
from enum import Enum


class audioSources(Enum):
    No_Audio = 0
    Device_Mic = 1  # Pupil Invisible microphone
    Screen_Audio = 2  # Screen Audio as defined by your recording method


def init_parser():
    parser = argparse.ArgumentParser(description="Pupil Labs - Dynamic RIM Module")
    parser.add_argument(
        "--screen_video_path", default=None, type=str, help="Path to the screen video"
    )
    parser.add_argument(
        "--raw_folder_path", default=None, type=str, help="Path to the raw folder"
    )
    parser.add_argument(
        "--rim_folder_path", default=None, type=str, help="Path to the RIM folder"
    )
    parser.add_argument(
        "--corners_screen",
        default=None,
        type=str,
        help="Path to the corners_screen file",
    )
    parser.add_argument(
        "--out_video_path",
        default=None,
        type=str,
        help="Path where to save the output video",
    )
    parser.add_argument(
        "--out_csv_path",
        default=None,
        type=str,
        help="Path where to save the output CSV",
    )
    parser.add_argument(
        "--audio",
        default="Device_Mic",
        choices=["No_Audio", "Device_Mic", "Screen_Audio"],
        type=str,
        help="Audio source, between device, computer, or none",
    )
    parser.add_argument(
        "--saveCSV", default=True, type=bool, help="Save the gaze data as a CSV file"
    )
    parser.add_argument(
        "--labels", default=True, type=bool, help="Show labels on the video"
    )
    parser.add_argument(
        "-p",
        "--visualise",
        action="store_true",
        help="Visualise the video as it creates",
    )
    parser.set_defaults(visualise=False)
    return parser
