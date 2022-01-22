from pathlib import Path

import depthai

for device in depthai.Device.getAllAvailableDevices():
    print(f"{device.getMxId()} {device.state}")

    with depthai.Device() as device:
        calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())
        calibData = device.readCalibration()

        calibData.eepromToJsonFile(calibFile)
