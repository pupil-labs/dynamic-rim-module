"""
Python 3.10 Dynamic RIM Script
This script is used to plot gaze over a video displayed on RIM
enrichment eye tracking recording.
"""

import glob
import logging

# Importing necessary libraries
import os
import struct
import time
from enum import Enum
from fractions import Fraction

import av
import cv2
import numpy as np
import pandas as pd

from pupil_labs.dynamic_content_on_rim.uitools.get_corners import pick_point_in_image
from pupil_labs.dynamic_content_on_rim.uitools.ui_tools import (
    get_file,
    get_path,
    get_savedir,
    progress_bar,
)
from pupil_labs.dynamic_content_on_rim.video.read import get_frame, read_video_ts

# Check if they are using a 64 bit architecture
verbit = struct.calcsize("P") * 8
if verbit != 64:
    error = "Sorry, this script only works on 64 bit systems!"
    raise Exception(error)

# Preparing the logger
logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("libav.swscaler").setLevel(logging.ERROR)


class audioSources(Enum):
    No_Audio = 0
    Pupil_Invisible_Mic = 1  # Pupil Invisible microphone
    Screen_Audio = 2  # Screen Audio as defined by your recording method


def main(
    sc_video_path=None,
    raw_data_path=None,
    rim_dir=None,
    corners_screen=None,
    out_vidpath=None,
    out_csvpath=None,
    audio=audioSources.Pupil_Invisible_Mic,
    _saveCSV=True,
    _labels=True,
):
    logging.info("Starting dynamic RIM script...")
    if audio.name not in audioSources._member_names_:
        error = f"Unknown audio option {audio}, use {audioSources._member_names_}"
        logging.error(error)
        raise Exception(error)
    # Ask the user to select the RIM folder if none is provided
    rim_dir = get_path("Select the RIM directory", "gaze.csv", rim_dir)
    start_time = time.time()
    # Read a pandas dataframe from the gaze file
    logging.info("Reading gaze data...")
    oftype = {"timestamp [ns]": np.uint64}
    gaze_rim_df = pd.read_csv(os.path.join(rim_dir, "gaze.csv"), dtype=oftype)
    sections_rim_df = pd.read_csv(os.path.join(rim_dir, "sections.csv"), dtype=oftype)
    gaze_rim_df = pd.merge(gaze_rim_df, sections_rim_df)

    # Read the world timestamps and gaze on ET
    logging.info("Reading world timestamps, events, and gaze on ET...")
    raw_data_path = get_path(
        "Select the video folder in the raw directory",
        "world_timestamps.csv",
        raw_data_path,
    )
    world_timestamps_df = pd.read_csv(
        os.path.join(raw_data_path, "world_timestamps.csv"), dtype=oftype
    )
    events_df = pd.read_csv(os.path.join(raw_data_path, "events.csv"), dtype=oftype)
    gaze_df = pd.read_csv(os.path.join(raw_data_path, "gaze.csv"), dtype=oftype)

    # Check if files belong to the same recording
    gaze_rim_df = check_ids(gaze_df, world_timestamps_df, gaze_rim_df)

    # Read the videos
    files = glob.glob(os.path.join(raw_data_path, "*.mp4"))
    if len(files) != 1:
        error = "There should be only one video in the raw folder!"
        logging.error(error)
        raise Exception(error)
    et_video_path = files[0]
    sc_video_path = get_file(sc_video_path)

    logging.info("Reading screen video...")
    _, sc_frames, sc_pts, sc_ts = read_video_ts(sc_video_path)

    logging.info("Reading eye tracking video...")
    _, et_frames, et_pts, et_ts = read_video_ts(et_video_path)
    if audio in audioSources:
        arguments = {"audio": True}
        if audio == audioSources.Pupil_Invisible_Mic:
            logging.info("Reading PI audio...")
            arguments["video_path"] = et_video_path
        elif audio == audioSources.Screen_Audio:
            logging.info("Reading screen audio...")
            arguments["video_path"] = sc_video_path
        _, audio_frames, audio_pts, audio_ts = read_video_ts(**arguments)

    logging.debug(
        "The screen video had a mean frame pts diff {} (SD: {})".format(
            np.mean(np.diff(sc_ts)), np.std(np.diff(sc_ts))
        )
    )
    logging.debug(
        "The Scene PI video had a mean frame pts diff {} (SD: {})".format(
            np.mean(np.diff(et_ts)), np.std(np.diff(et_ts))
        )
    )

    # Select corners of the screen
    logging.info("Select corners of the screen...")
    corners_screen, ref_img = pick_point_in_image(rim_dir, corners_screen, 4)

    # Compute the perspective transform
    logging.info("Computing the perspective transform...")
    M = get_perspective_transform(corners_screen, ref_img, sc_video_path, False)
    # Transform the gaze points using M
    logging.info("Transforming the gaze points using M...")
    # Preparing data to transform
    xy = np.expand_dims(
        gaze_rim_df[
            [
                "gaze position in reference image x [px]",
                "gaze position in reference image y [px]",
            ]
        ].to_numpy(dtype=np.float32),
        axis=0,
    )
    # Applying the perspective transform
    xy_transf = cv2.perspectiveTransform(xy, M)
    # Saving the transformed gaze points
    gaze_rim_df[
        ["gaze position transf x [px]", "gaze position transf y [px]"]
    ] = pd.DataFrame(xy_transf[0]).set_index(gaze_rim_df.index)
    # Get the patch of the screen
    mask = np.zeros(np.asarray(ref_img.shape)[0:2], dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [
            np.stack(
                (
                    corners_screen["upper left"],
                    corners_screen["upper right"],
                    corners_screen["lower right"],
                    corners_screen["lower left"],
                )
            )
        ],
        (255),
    )
    _, timg = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    _screen, _ = cv2.findContours(timg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Timestamp matching
    logging.info("Timestamp matching...")
    start_video_ns = events_df.loc[events_df["name"] == "start.video"][
        "timestamp [ns]"
    ].values[0]
    # Create some timestamps [ns] for the screen video to match, use the start_video_ns
    # and the ts of the screen video
    sc_timestamps_ns = sc_ts + start_video_ns
    if audio == audioSources.Screen_Audio:
        audio_ts = audio_ts + start_video_ns
    end_video_ns = np.min(
        [np.max(sc_timestamps_ns), np.max(world_timestamps_df["timestamp [ns]"])]
    )
    # Match the timestamps gaze_rim_df, world_timestamps_df, gaze_df, fake_timestamps_ns
    sc_video_df = pd.DataFrame()
    sc_video_df["frame"] = np.arange(sc_frames)
    sc_video_df["timestamp [ns]"] = sc_timestamps_ns
    sc_video_df["pts"] = [int(pt) for pt in sc_pts]

    if audio in audioSources and audio is not audioSources.No_Audio:
        audio_df = pd.DataFrame()
        audio_df["frame"] = np.arange(audio_frames)
        audio_df["timestamp [ns]"] = np.array(audio_ts)
        audio_df["pts"] = [int(pt) for pt in audio_pts]

    et_video_df = pd.DataFrame()
    et_video_df["frame"] = np.arange(et_frames)
    et_video_df["timestamp [ns]"] = world_timestamps_df["timestamp [ns]"]
    et_video_df["pts"] = [int(pt) for pt in et_pts]

    logging.info("Creating matched timestamps tables for video:")
    merged = merge_tables(
        sc_video_df,
        None,
        et_video_df,
        gaze_rim_df.sort_values(by=["timestamp [ns]"]),
        gaze_df,
        "_video",
        "_audio",
    )
    merged = merged[merged["timestamp [ns]"] <= end_video_ns]

    if audio in audioSources and audio != audioSources.No_Audio:
        logging.info("Creating matched timestamps tables for audio:")
        merged_audio = merge_tables(
            audio_df,
            sc_video_df,
            et_video_df,
            gaze_rim_df.sort_values(by=["timestamp [ns]"]),
            gaze_df,
            "_audio",
            "_video",
        )
        merged_audio = merged_audio[merged_audio["timestamp [ns]"] <= end_video_ns]
    else:
        merged_audio = None

    logging.info(merged.describe().transpose())

    # Plot the videos
    logging.info("Plotting/Recording...")
    save_videos(
        corners_screen,
        merged,
        ref_img,
        _screen,
        sc_video_path,
        et_video_path,
        out_vidpath,
        merged_audio,
        audio,
        _labels,
    )
    if _saveCSV:
        logging.info("Saving CSV...")
        gaze_rim_df.to_csv(get_savedir(out_csvpath, "csv"), index=True)
    logging.info("Mischief managed! ⚡️")
    logging.info("Executed in: %s seconds" % (time.time() - start_time))


def merge_tables(table1, table2, table3, table4, table5, label1, label2):
    logging.info("Merging audio and image for screen video...")
    for i, tbl in enumerate([table1, table2, table3, table4, table5]):
        if tbl is not None and tbl["timestamp [ns]"].dtype != (np.uint64):
            error = "Table {} has a timestamp column with the wrong dtype: {}".format(
                i,
                tbl["timestamp [ns]"].dtype,
            )
            logging.error(error)
            raise ValueError(error)
    if table2 is not None:
        merged_sc = pd.merge_asof(
            table1,
            table2,
            on="timestamp [ns]",
            direction="nearest",
            suffixes=[label1, label2],
        )
    else:
        merged_sc = table1
    logging.info("Merging videos timestamps...")
    merged_vid = pd.merge_asof(
        merged_sc,
        table3,
        on="timestamp [ns]",
        direction="nearest",
        suffixes=["_sc", "_world"],
    )
    logging.info("Merging gaze timestamps...")
    merged_gaze = pd.merge_asof(
        table4,
        table5,
        on="timestamp [ns]",
        direction="nearest",
        suffixes=["_rim", "_orig"],
    )
    logging.info("Merging all together...")
    merged = pd.merge_asof(
        merged_vid, merged_gaze, on="timestamp [ns]", suffixes=["_vid", "_gaze"]
    )
    return merged


def get_perspective_transform(corners_screen, ref_img, sc_video_path, debug=False):
    """
    A function to get the perspective transform between the screen
    and the reference image using OpenCV
    :param corners_screen: corners of the screen coordinates
    :param ref_img: reference image
    :param sc_video_path: path to the screen video
    """
    pnts1 = np.float32(
        [
            corners_screen["upper left"],
            corners_screen["lower left"],
            corners_screen["upper right"],
            corners_screen["lower right"],
        ]
    )
    if debug:
        img = ref_img
    else:
        with av.open(sc_video_path) as video_container:
            img = next(video_container.decode(video=0)).to_image()
    pnts2 = np.float32(
        [
            [0, 0],  # upper left
            [0, img.height],  # lower left
            [img.width, 0],  # upper right
            [img.width, img.height],  # lower right
        ]
    )
    # Get the transformation matrix
    M = cv2.getPerspectiveTransform(pnts1, pnts2)
    if debug:
        logging.info("Transformation matrix:")
        logging.info(M)
        cv2.imshow(
            "TMat",
            cv2.warpPerspective(ref_img, M, (ref_img.shape[1], ref_img.shape[0])),
        )

        cv2.waitKey()
    return M


def save_videos(  # noqa: C901 Ignore `Function too complex` flake8 error. TODO: Fix
    corners_screen,
    merged,
    ref_img,
    _screen,
    sc_video_path,
    et_video_path,
    out_vidpath,
    merged_audio,
    audio,
    _labels=True,
    _visualise=False,
    _recording=True,
):
    """
    A function to plot/record the ref image, eye tracking video and
    the screen recorded video with a gaze overlay, as a cv2.
    :param corners_screen: The corners of the screen in the reference image
    :param merged: The merged dataframe
    :param ref_img: The reference image
    :param _screen: The screen contour in openCV format
    :param sc_video_path: The screen recorded video path
    :param et_video_path: The eye tracking video path
    :param _visualise: If True, the video will be visualised
    :param _recording: If True, the video will be recorded
    :param _labels: If True, the videos will be labelled
    """
    corners_screen = np.stack(
        (
            corners_screen["upper left"],
            corners_screen["upper right"],
            corners_screen["lower right"],
            corners_screen["lower left"],
        )
    )
    # Decode the first frames and read the max height and width of the videos
    with av.open(et_video_path) as et_video, av.open(sc_video_path) as sc_video:
        _etframe = next(et_video.decode(video=0))
        mheight = _etframe.height
        mwidth = _etframe.width
        _scframe = next(sc_video.decode(video=0))
        if merged_audio is not None:
            if audio == audioSources.Pupil_Invisible_Mic:
                audio_frame = next(et_video.decode(audio=0))
            elif audio == audioSources.Screen_Audio:
                audio_frame = next(sc_video.decode(audio=0))
        if _scframe.height > mheight:
            mheight = _scframe.height
        if _scframe.width > mwidth:
            mwidth = _scframe.width

    refimg_finalwidth = int(ref_img.shape[1] * mheight / ref_img.shape[0])

    # Np array to hold the frames
    bkg = np.zeros((mheight, _etframe.width + _scframe.width + refimg_finalwidth, 3))

    # Locate images' origins on bkg
    c_refimg = [int(0), int(_etframe.width)]
    c_etvid = [
        int((mheight / 2) - (_etframe.height / 2)),
        int(0),
    ]
    c_scvid = [
        int((mheight / 2) - (_scframe.height / 2)),
        int(_etframe.width + refimg_finalwidth),
    ]

    # Select where to store the video
    out_vidpath = get_savedir(out_vidpath, "video")

    with av.open(et_video_path) as et_video, av.open(
        sc_video_path
    ) as sc_video, av.open(sc_video_path) as sc_audio, av.open(
        et_video_path
    ) as et_audio, av.open(
        out_vidpath, "w"
    ) as out_container:
        out_container.metadata["title"] = "Merged video"
        out_video = out_container.add_stream("libx264", rate=30, options={"crf": "18"})
        out_video.height = mheight
        out_video.width = _etframe.width + _scframe.width + refimg_finalwidth
        out_video.pix_fmt = "yuv420p"
        out_video.codec_context.time_base = Fraction(1, 30)
        if audio in audioSources and merged_audio is not None:
            audio_arguments = {"codec_name": "aac", "layout": "stereo"}
            if audio == audioSources.Pupil_Invisible_Mic:
                audio_arguments["rate"] = et_audio.streams.audio[0].rate
            elif audio == audioSources.Screen_Audio:
                audio_arguments["rate"] = sc_audio.streams.audio[0].rate
            out_audio = out_container.add_stream(**audio_arguments)
        out_audio.time_base = out_audio.codec_context.time_base
        idx = 0
        _etlpts = -1
        _sclpts = -1
        while idx < merged.shape[0]:
            row = merged.iloc[idx]
            _etframe, _etlpts = get_frame(
                et_video, int(row["pts_world"]), _etlpts, _etframe
            )
            _scframe, _sclpts = get_frame(
                sc_video, int(row["pts_sc"]), _sclpts, _scframe
            )

            if _etframe is None or _scframe is None:
                break

            ref_img_final = prepare_image(
                ref_img,
                row[
                    [
                        "gaze position in reference image x [px]",
                        "gaze position in reference image y [px]",
                    ]
                ],
                "Reference Image",
                corners_screen,
                _screen,
                mheight,
            )
            _etframe_final = prepare_image(
                _etframe.to_image().convert("RGB"),
                row[["gaze x [px]", "gaze y [px]"]],
                "PI Video",
                corners_screen,
                _screen,
            )
            _scframe_final = prepare_image(
                _scframe.to_image().convert("RGB"),
                row[["gaze position transf x [px]", "gaze position transf y [px]"]],
                "Screen Video",
                corners_screen,
                _screen,
            )

            # Add images to bkg
            bkg[
                c_refimg[0] : c_refimg[0] + ref_img_final.shape[0],
                c_refimg[1] : c_refimg[1] + ref_img_final.shape[1],
                :,
            ] = ref_img_final
            bkg[
                c_etvid[0] : c_etvid[0] + _etframe_final.shape[0],
                c_etvid[1] : c_etvid[1] + _etframe_final.shape[1],
                :,
            ] = _etframe_final
            bkg[
                c_scvid[0] : c_scvid[0] + _scframe_final.shape[0],
                c_scvid[1] : c_scvid[1] + _scframe_final.shape[1],
                :,
            ] = _scframe_final
            if _labels:
                # Add text to the frames
                y, w, h = 60, 360, 60
                labels = [
                    {"text": "PI Video", "margin": 10},
                    {"text": "Reference Image", "margin": 10 + _etframe.width},
                    {
                        "text": "Screen Video",
                        "margin": 10 + (_etframe.width + refimg_finalwidth),
                    },
                ]
                for label in labels:
                    overlay = bkg[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10 + w)
                    ]
                    whiterect = np.ones(overlay.shape) * 255
                    res = cv2.addWeighted(overlay, 0.5, whiterect, 0.5, 0)
                    bkg[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10) + w
                    ] = res
                    bkg = cv2.putText(
                        bkg,
                        label["text"],
                        (label["margin"], 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        2,
                        2,
                    )
            # Get the final frame
            out_ = bkg.copy()
            out_ = cv2.normalize(out_, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if _visualise:
                cv2.imshow("Merged Video", out_)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            if _recording:
                # Convert to av frame
                cv2.cvtColor(out_, cv2.COLOR_BGR2RGB, out_)
                np.expand_dims(out_, axis=2)
                out_frame = av.VideoFrame.from_ndarray(out_, format="rgb24")
                for packet in out_video.encode(out_frame):
                    out_container.mux(packet)

                progress_bar(idx, merged.shape[0], "encoding video")
            idx += 1
        # Flush the encoder
        if _recording:
            for packet in out_video.encode(None):
                out_container.mux(packet)
        if merged_audio is not None and _recording:
            idx = 0
            audio_pts = -1
            while idx < merged_audio.shape[0]:
                row = merged_audio.iloc[idx]
                if audio == audioSources.Pupil_Invisible_Mic:
                    audio_frame, audio_pts = get_frame(
                        et_audio,
                        int(row["pts_audio"]),
                        audio_pts,
                        audio_frame,
                        audio=True,
                    )
                elif audio == audioSources.Screen_Audio:
                    audio_frame, audio_pts = get_frame(
                        sc_audio,
                        int(row["pts_audio"]),
                        audio_pts,
                        audio_frame,
                        audio=True,
                    )
                logging.debug(f"Audio frame time: {audio_frame.time}s")
                if audio_frame is None:
                    break
                elif audio_frame.is_corrupt:
                    logging.info(f"Frame {idx} is corrupt")
                else:
                    audio_frame.pts = (
                        None  # https://github.com/PyAV-Org/PyAV/issues/761
                    )
                    aframes = out_audio.encode(audio_frame)
                    out_container.mux(aframes)

                progress_bar(idx, merged_audio.shape[0], label=" encoding audio")
                idx += 1
            # After loop finished flush
            for packet in out_audio.encode(None):
                out_container.mux(packet)
        if _recording:
            out_container.close()
        logging.info("Video saved to: " + out_vidpath)


def prepare_image(frame, xy, str, corners_screen, _screen, mheight=0, alpha=0.3):
    """
    Prepares an image for merging with the background.
    Adds a circle to the image where the gaze is located.
    :param frame: The image to be merged.
    :param xy: The gaze position in the image.
    :param str: The name of the image.
    :param corners_screen: The corners of the screen.
    :param _screen: Screen contour in openCV format.
    :param mheight: The height of the merged video.
    :param alpha: The transparency of the screen overlay.
    """
    frame = np.asarray(frame, dtype=np.float32)
    frame = frame[:, :, :]
    xy = xy.to_numpy(dtype=np.int32)
    # Frame to bgr
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Add screen overlay and downsize ref image
    if str == "Reference Image":
        if cv2.pointPolygonTest(_screen[0], (int(xy[0]), int(xy[1])), False) == 1:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]
        resizefactor = frame.shape[0] / mheight
        # Screen overlay
        roi = (
            slice(
                int(np.floor(corners_screen[:, 1].min())),
                int(np.ceil(corners_screen[:, 1].max())),
            ),
            slice(
                int(np.floor(corners_screen[:, 0].min())),
                int(np.ceil(corners_screen[:, 0].max())),
            ),
        )
        overlay = frame.copy()
        cv2.fillPoly(frame, [(corners_screen.astype(int))], color)
        cv2.addWeighted(frame[roi], alpha, overlay[roi], 1 - alpha, 0, frame[roi])
        # Gazepoint overlay
        frame = cv2.circle(
            frame, xy, int(50 * resizefactor), (0, 0, 255), int(10 * resizefactor)
        )
        # Downsize ref image
        frame = cv2.resize(
            frame, (int(frame.shape[1] * mheight / frame.shape[0]), int(mheight))
        )  # col, rows
    elif str == "PI Video":
        resizefactor = 2 / 3
        frame = cv2.circle(
            frame, xy, int(50 * resizefactor), (0, 0, 255), int(10 * resizefactor)
        )
    elif str == "Screen Video":
        resizefactor = 1
        frame = cv2.circle(
            frame, xy, int(50 * resizefactor), (0, 0, 255), int(10 * resizefactor)
        )
    return frame


def check_ids(gaze_df, world_timestamps_df, gaze_rim_df):
    """
    Checks if the recording IDs of the gaze data and the world timestamps are the same.
    :param gaze_df: The gaze data.
    :param world_timestamps_df: The world timestamps.
    :param gaze_rim_df: The gaze data from the RIM.
    returns gaze_rim_df with only the matching ID.
    """
    g_ids = gaze_df["recording id"].unique()
    w_ids = world_timestamps_df["recording id"].unique()
    rim_ids = gaze_rim_df["recording id"].unique()
    if g_ids.shape[0] != 1 or w_ids.shape[0] != 1:
        error_base = f"None or more than one recording ID found "
        if g_ids.shape[0] != 1:
            error_end = "in gaze data: {g_ids}"
        elif w_ids.shape[0] != 1:
            error_end = "in world timestamps: {w_ids}"
        logging.error(error_base + error_end)
        raise SystemExit(error_base + error_end)
    if not np.isin(rim_ids, g_ids).any():
        error = (
            "Recording ID of RIM gaze data does not match recording ID"
            " of the Raw data, please check if you selected the"
            " right folder."
        )
        logging.error(error)
        raise SystemExit(error)
    else:
        ID = g_ids[0]
        logging.info(
            f"""Recording ID of RIM gaze data matches recording ID of the RAW data
            id: {ID} """
        )
        isID = gaze_rim_df["recording id"] == ID
        gaze_rim_df.drop(gaze_rim_df.loc[np.invert(isID)].index, inplace=True)
        # removing NaNs from gaze rim data
        gaze_rim_df.dropna(
            inplace=True,
            subset=[
                "gaze position in reference image x [px]",
                "gaze position in reference image y [px]",
            ],
        )
        if gaze_rim_df.empty:
            error = f"No valid gaze data in RIM gaze data for recording ID {ID}"
            logging.error(error)
            raise SystemExit(error)
        logging.info(gaze_rim_df.head())
    return gaze_rim_df


if __name__ == "__main__":
    main()
