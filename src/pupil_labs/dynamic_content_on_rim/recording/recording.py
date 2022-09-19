import asyncio
import logging
import os
import platform
import time

import numpy as np
import simpleobsws
from pupil_labs.realtime_api import Device, Network, StatusUpdateNotifier
from pupil_labs.realtime_api.models import Recording
from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator

logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def print_recording(component):
    if isinstance(component, Recording):
        logging.info(f"Update: {component.message}")


async def main():
    async with Network() as network:
        dev_info = await network.wait_for_new_device(timeout_seconds=5)
    if dev_info is None:
        print("No device could be found! Abort")
        return

    async with Device.from_discovered_device(dev_info) as device:
        status = await device.get_status()

        print(f"Device IP address: {status.phone.ip}")
        print(f"Device Time Echo port: {status.phone.time_echo_port}")
        print(f"Battery level: {status.phone.battery_level} %")

        print(f"Connected glasses: SN {status.hardware.glasses_serial}")
        print(f"Connected scene camera: SN {status.hardware.world_camera_serial}")

        world = status.direct_world_sensor()
        print(f"World sensor: connected={world.connected} url={world.url}")

        gaze = status.direct_gaze_sensor()
        print(f"Gaze sensor: connected={gaze.connected} url={gaze.url}")

        time_offset_estimator = TimeOffsetEstimator(
            status.phone.ip, status.phone.time_echo_port
        )
        estimated_offset = await time_offset_estimator.estimate()
        logging.info(f"Estimated time offset: {estimated_offset} ms")

        notifier = StatusUpdateNotifier(device, callbacks=[print_recording])
        await notifier.receive_updates_start()
        recording_id = await device.recording_start()
        logging.info(f"Initiated recording with id {recording_id}")
        await notifier.receive_updates_stop()

        # Open OBS
        if platform.system() == "Windows":
            startOBScmd = "start OBS.exe"
        elif platform.system() == "Darwin":
            startOBScmd = "open -a OBS.app"
        else:
            startOBScmd = "./OBS"
        logging.info(f"Starting OBS with command: {startOBScmd}")
        os.system(startOBScmd)

        # wait 5s for OBS to start
        await asyncio.sleep(5)

        parameters = simpleobsws.IdentificationParameters(
            ignoreNonFatalRequestChecks=False
        )
        ws = simpleobsws.WebSocketClient(
            url="ws://localhost:4455/",
            password="y3Xq6EKRJD5fnbaG",
            identification_parameters=parameters,
        )
        offset = estimated_offset.time_offset_ms.mean * 1e6
        await ws.connect()
        await ws.wait_until_identified()
        logging.info("Connected and identified in OBS-websocket")

        requests = simpleobsws.Request("StartRecord")
        befreq = time.time_ns()
        ret = await ws.call(requests)  # Perform the request
        aftereq = time.time_ns()
        if ret.ok():  # Check if the request succeeded
            logging.info(f"Request succeeded! Response data: {ret.responseData}")
            logging.info("Screen recording started")
            logging.info(
                await device.send_event(
                    "start.video",
                    event_timestamp_unix_ns=np.mean([aftereq, befreq]) - offset,
                )
            )

        await ws.disconnect()
        logging.info("Disconnected from OBS-websocket")


if __name__ == "__main__":
    asyncio.run(main())
