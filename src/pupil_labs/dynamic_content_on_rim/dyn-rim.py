"""
Python 3.10 Dynamic RIM Script
This script is used to plot gaze over a video displayed on RIM enrichmed eye tracking recording.
First, ensure you have the required dependencies installed:
    pip install -r requirements.txt
Then run the script:
    python dyn-rim.py
@author: mgg
Version 1.0
Date: 04/08/2022
"""

import logging

# Importing necessary libraries
import os
import pathlib
import time
import tkinter as tk
import uuid
from tkinter import filedialog

import av
import cv2
import numpy as np
import pandas as pd

# Preparing the logger
logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("libav.swscaler").setLevel(logging.ERROR)


def main(withAudio=True):
    # Ask the user to select the RIM folder
    rim_dir = get_path("Select the RIM directory", "gaze.csv")
    start_time = time.time()
    # Read a pandas dataframe from the gaze file
    logging.info("Reading gaze data...")
    gaze_rim_df = pd.read_csv(os.path.join(rim_dir, "gaze.csv"))
    sections_rim_df = pd.read_csv(os.path.join(rim_dir, "sections.csv"))
    gaze_rim_df = pd.merge(gaze_rim_df, sections_rim_df)
    gaze_rim_df.dropna(
        inplace=True,
        subset=[
            "gaze position in reference image x [px]",
            "gaze position in reference image y [px]",
        ],
    )
    logging.info(gaze_rim_df.head())

    # Read the world timestamps and gaze on ET
    logging.info("Reading world timestamps, events, and gaze on ET...")
    raw_data_path = get_path(
        "Select the video folder in the raw directory", "world_timestamps.csv"
    )
    world_timestamps_df = pd.read_csv(
        os.path.join(raw_data_path, "world_timestamps.csv")
    )
    events_df = pd.read_csv(os.path.join(raw_data_path, "events.csv"))
    gaze_df = pd.read_csv(os.path.join(raw_data_path, "gaze.csv"))

    # Read the videos
    root = tk.Tk()
    root.withdraw()
    sc_video_path = filedialog.askopenfilename(
        title="Select the screen video",
        filetypes=[("Video files", "*.mp4 *.mkv *.avi")],
    )
    et_video_path = filedialog.askopenfilename(
        title="Select the ET video",
        initialdir=raw_data_path,
        filetypes=[("Video files", "*.mp4")],
    )

    logging.info("Reading screen video...")
    _, sc_frames, sc_pts, sc_ts = read_screen_video(sc_video_path)
    if withAudio:
        logging.info("Reading screen audio...")
        _, asc_frames, asc_pts, asc_ts = read_screen_video(sc_video_path, audio=True)
    logging.info("Reading eye tracking video...")
    _, et_frames, et_pts, et_ts = read_screen_video(et_video_path)

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
    corners_screen, ref_img = pick_point_in_image(rim_dir, 4)
    # Reorder the corners
    corners_screen = np.array(corners_screen)
    corners_screen = corners_screen[np.argsort(corners_screen[:, 0])]
    corners_screen = corners_screen[np.argsort(corners_screen[:, 1])]
    corners_screen = corners_screen[[0, 1, 3, 2], :]
    logging.info(corners_screen)

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
        ].to_numpy(dtype=np.float64),
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
    cv2.fillPoly(mask, [corners_screen], (255))
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
    if withAudio:
        asc_timestamps_ns = asc_ts + start_video_ns
    end_video_ns = np.min(
        [np.max(sc_timestamps_ns), np.max(world_timestamps_df["timestamp [ns]"])]
    )
    # Match the timestamps gaze_rim_df, world_timestamps_df, gaze_df, fake_timestamps_ns
    sc_video_df = pd.DataFrame()
    sc_video_df["frame"] = np.arange(sc_frames)
    sc_video_df["timestamp [ns]"] = sc_timestamps_ns.astype(int)
    sc_video_df["pts"] = [int(pt) for pt in sc_pts]
    if withAudio:
        sc_audio_df = pd.DataFrame()
        sc_audio_df["frame"] = np.arange(asc_frames)
        sc_audio_df["timestamp [ns]"] = asc_timestamps_ns.astype(int)
        sc_audio_df["pts"] = [int(pt) for pt in asc_pts]

    et_video_df = pd.DataFrame()
    et_video_df["frame"] = np.arange(et_frames)
    et_video_df["timestamp [ns]"] = world_timestamps_df["timestamp [ns]"]
    et_video_df["pts"] = [int(pt) for pt in et_pts]

    logging.info("Creating matched timestamps tables for video:")
    merged = merge_tables(
        sc_video_df,
        sc_audio_df,
        et_video_df,
        gaze_rim_df.sort_values(by=["timestamp [ns]"]),
        gaze_df,
        "_video",
        "_audio",
    )
    merged = merged[merged["timestamp [ns]"] <= end_video_ns]

    if withAudio:
        logging.info("Creating matched timestamps tables for video:")
        merged_audio = merge_tables(
            sc_audio_df,
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
        merged_audio,
    )

    logging.info("Mischief managed! ⚡️")
    logging.info("Executed in: %s seconds" % (time.time() - start_time))


def merge_tables(table1, table2, table3, table4, table5, label1, label2):
    logging.info("Merging audio and image for screen video...")
    merged_sc = pd.merge_asof(
        table1,
        table2,
        on="timestamp [ns]",
        direction="nearest",
        suffixes=[label1, label2],
    )
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


def read_screen_video(video_path, audio=False):
    """
    A function to read a video and return the fps,
    the number of frames, the pts and timestamps.
    :param video_path: the path to the video
    """
    nframes = []
    # Read the video
    with av.open(video_path) as video_container:
        if audio:
            stream = video_container.streams.audio[0]
        else:
            stream = video_container.streams.video[0]
        fps = stream.average_rate  # alt base_rate or guessed_rate
        nframes = stream.frames
        logging.info("Extracting pts...")
        pts = []
        ts = []
        for packet in video_container.demux(stream):
            for frame in packet.decode():
                if frame is not None and frame.pts is not None:
                    pts.append(frame.pts)
                    ts.append(
                        float(
                            frame.pts * frame.time_base
                            - stream.start_time * frame.time_base
                        )
                        * 1e9
                    )
        pts.sort()
        ts.sort()

        if nframes != len(pts):
            nframes = len(pts)
        else:
            logging.info(f"Video has {nframes} frames")
    return fps, nframes, pts, ts


def pick_point_in_image(rim_dir, npoints=4):
    """
    A function to pick the screen corners on the reference image,
    returns the corners points, and the ref image in openCV
    :param rim_dir: the directory of the reference image
    :param npoints: the number of corners to pick
    """
    # Pick the image
    image_path = os.path.join(rim_dir, "reference_image.jpeg")
    # Read the image
    image = cv2.imread(image_path)
    copy_image = image.copy()
    points = []

    def pick_corners(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < npoints:
                cv2.circle(param, (x, y), 50, (0, 0, 255), -1)
                cv2.circle(image, (x, y), 50, (0, 0, 255), -1)
                points.append((x, y))
                logging.info(f"Picked point: {(x, y)}")
            else:
                cv2.putText(
                    param,
                    "Done, press Q to continue",
                    (1000, 1000),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5,
                    (0, 0, 255),
                    15,
                    2,
                )

    cv2.namedWindow("Pick the corners of your ROI with double clicks")
    cv2.setMouseCallback(
        "Pick the corners of your ROI with double clicks", pick_corners, copy_image
    )
    while True:
        cv2.imshow("Pick the corners of your ROI with double clicks", copy_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return points, image


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
            corners_screen[0, :],  # upper left
            corners_screen[3, :],  # lower left
            corners_screen[1, :],  # upper right
            corners_screen[2, :],  # lower right
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
            cv2.warpPerspective(ref_img, M, (ref_img.shape[1], ref_img.shape[0]))
        )
        cv2.waitKey()
    return M


def save_videos(
    corners_screen,
    merged,
    ref_img,
    _screen,
    sc_video_path,
    et_video_path,
    merged_audio,
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
    """
    # Decode the first frames and read the max height and width of the videos
    with av.open(et_video_path) as et_video, av.open(sc_video_path) as sc_video:
        _etframe = next(et_video.decode(video=0))
        mheight = _etframe.height
        mwidth = _etframe.width
        _scframe = next(sc_video.decode(video=0))
        if merged_audio is not None:
            a_scframe = next(sc_video.decode(audio=0))
        if _scframe.height > mheight:
            mheight = _scframe.height
        if _scframe.width > mwidth:
            mwidth = _scframe.width

    refimg_finalwidth = int(ref_img.shape[1] * mheight / ref_img.shape[0])

    # Np array to hold the frames
    bkg = np.zeros((mheight, _etframe.width + _scframe.width + refimg_finalwidth, 3))

    # Locate images' origins on bkg
    c_refimg = [int(0), int(mwidth)]
    c_etvid = [
        int((mheight / 2) - (_etframe.height / 2)),
        int((mwidth / 2) - (_etframe.width / 2)),
    ]
    c_scvid = [
        int((mheight / 2) - (_scframe.height / 2)),
        int(mwidth + refimg_finalwidth),
    ]

    # Select where to store the video
    out_vidpath = filedialog.asksaveasfilename(
        defaultextension=".mp4",
        initialdir=pathlib.Path.home(),
        initialfile=(str(uuid.uuid4()) + ".mp4"),
        title="Select where to save the output video",
    )

    with av.open(et_video_path) as et_video, av.open(
        sc_video_path
    ) as sc_video, av.open(sc_video_path) as sc_audio, av.open(
        out_vidpath, "w"
    ) as out_container:
        out_container.metadata["title"] = "Merged video"
        out_video = out_container.add_stream("libx264", rate=30, options={"crf": "18"})
        out_video.height = mheight
        out_video.width = _etframe.width + _scframe.width + refimg_finalwidth
        out_video.pix_fmt = "yuv420p"
        out_audio = out_container.add_stream(
            "aac", rate=sc_audio.streams.audio[0].rate, layout="stereo"
        )
        idx = 0
        _etlpts = -1
        _sclpts = -1
        while idx < merged.shape[0]:
            row = merged.iloc[idx]
            _etframe, _etlpts = get_frame(et_video, int(row["pts"]), _etlpts, _etframe)
            _scframe, _sclpts = get_frame(
                sc_video, int(row["pts_video"]), _sclpts, _scframe
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

            # Add text to the frames
            labels = [
                {"text": "PI Video", "margin": 10},
                {"text": "Reference Image", "margin": 10 + mwidth},
                {"text": "Screen Video", "margin": 10 + (mwidth + refimg_finalwidth)},
            ]
            for label in labels:
                bkg = cv2.putText(
                    bkg,
                    label["text"],
                    (label["margin"], 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    1,
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
        if merged_audio is not None:
            idx = 0
            a_sclpts = -1
            while idx < merged_audio.shape[0]:
                row = merged_audio.iloc[idx]
                a_scframe, a_sclpts = get_frame(
                    sc_audio, int(row["pts_audio"]), a_sclpts, a_scframe, audio=True
                )
                if a_scframe is None:
                    break
                for packet in out_audio.encode(a_scframe):
                    out_container.mux(packet)

                progress_bar(idx, merged_audio.shape[0], label=" encoding audio")
                idx += 1
        # After loop finished
        if _recording:
            for packet in out_audio.encode(None):
                out_container.mux(packet)
            for packet in out_video.encode(None):
                out_container.mux(packet)

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
        cv2.fillPoly(frame, [np.int32(corners_screen)], color)
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


def get_frame(av_container, pts, last_pts, frame, audio=False):
    """
    Gets the frame at the given timestamp.
    :param av_container: The container of the video.
    :param pts: The pts of the frame we are looking for.
    :param last_pts: The last pts of the video readed.
    :param frame: Last frame decoded.
    """
    if audio:
        strm = av_container.streams.audio[0]
    else:
        strm = av_container.streams.video[0]
    if last_pts < pts:
        try:
            for frame in av_container.decode(strm):
                logging.debug(
                    f"Frame {frame.pts} read from video and looking for {pts}"
                )
                if pts == frame.pts:
                    last_pts = frame.pts
                    return frame, last_pts
                if pts < frame.pts:
                    logging.warning(f"Frame {pts} not found in video, used {frame.pts}")
                    last_pts = frame.pts
                    return frame, last_pts
        except av.EOFError:
            logging.info("End of the file")
            return None, last_pts
    else:
        logging.debug("This frame was already decoded")
        return frame, last_pts


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


def get_path(msg, file):
    root = tk.Tk()
    root.withdraw()
    _path = filedialog.askdirectory(title=msg, message=msg)
    assert _path is not None, "No path selected"
    if not os.path.exists(_path):
        logging.error("File not found, probably wrong folder")
    return _path


if __name__ == "__main__":
    main()
