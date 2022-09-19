import logging

import av
import numpy as np


def read_video_ts(video_path, audio=False):
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
        pts, dts, ts = np.empty([3, 0, 0], dtype=np.uint64)
        for packet in video_container.demux(stream):
            for frame in packet.decode():
                if frame is not None and frame.pts is not None:
                    pts = np.append(pts, np.uint64(frame.pts))
                    dts = (
                        np.append(dts, np.uint64(frame.dts))
                        if frame.dts is not None
                        else logging.info(
                            f"Decoding timestamp is missing at frame {len(pts)}"
                        )
                    )
                    ts = np.append(
                        ts,
                        np.uint64(
                            (
                                frame.pts * frame.time_base
                                - stream.start_time * frame.time_base
                            )
                            * 1e9
                        ),
                    )
        if not isMonotonicInc(pts):
            logging.info("Pts are not monotonic increasing!.")
        if np.array_equal(pts, dts):
            logging.info("Pts and dts are equal, using pts")

        pts.sort()
        ts.sort()

        if nframes != len(pts):
            nframes = len(pts)
        else:
            logging.info(f"Video has {nframes} frames")
    return fps, nframes, pts, ts


def isMonotonicInc(a2check):
    return all(a2check[i] <= a2check[i + 1] for i in range(len(a2check) - 1))


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
