import depthai as dai
import numpy as np
from datetime import timedelta
from typing import Optional, Tuple, Dict

# Camera Settings
FPS: int = 15

class DepthAIPipeline:
    """
    This class creates a DepthAI Pipeline for the OAK-D Pro Device. 

    Important:
        - Depth units are returned in mm. Remember to convert to meters if needed.
    """

    def __init__(self):
        """
        Initialize the DepthAIPipeline instance.
        """
        # The DepthAI pipeline instance.
        self.pipeline = None
        # The DepthAI device instance.
        self.device = None

        # Output queues for color and depth frames.
        self.synced_queue = None

        # Output for IMU data.
        self.imu_queue = None
        

    def create_pipeline(self):
        """
        Create and return a DepthAI pipeline that streams both undistorted color (RGB)
        and depth frames.

        Returns:
            dai.Pipeline: Configured pipeline object.
        """
        # Create a new pipeline instance.
        pipeline = dai.Pipeline()

        # ---------------------------
        # Color
        # ---------------------------
        # 1) Create a generic camera object.
        cam = pipeline.create(dai.node.Camera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setSize(1280, 800)
        cam.setFps(FPS)
        cam.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
        cam.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT)
        
        # ---------------------------
        # DEPTH
        # ---------------------------
        # 1) Create the stereo depth node.
        stereo = pipeline.create(dai.node.StereoDepth)

        # 2) Create the mono cameras to be used for depth calculation.
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)

        # 3) Assign the left and right board sockets and set the camera resolution.
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        
        mono_left.setFps(FPS)
        mono_right.setFps(FPS)

        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B) 
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        # 4) Configure and link the cameras to the the depth node.
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        stereo.setSubpixel(True) # Improve precision at the cost of performance.
        stereo.setSubpixelFractionalBits(5)
        stereo.setExtendedDisparity(True) # Extended disparity increases the range of distances that can be measured.

        stereo.enableDistortionCorrection(True) # Enable on-device distortion correction for the stereo pair.
        stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
        stereo.setInputResolution(1280, 800)
        stereo.setOutputSize(1280, 800)

        # ---------------------------
        # OUTPUT
        # ---------------------------
        # Create the output objects for the pipeline.
        cfg = stereo.initialConfig.get()

        # Cost-matching confidence threshold:
        # Lower threshold -> stricter -> fewer outliers but more invalid pixels.
        cfg.costMatching.confidenceThreshold = 200

        # Speckle removal (general cleanup)
        cfg.postProcessing.speckleFilter.enable = True
        cfg.postProcessing.speckleFilter.speckleRange = 60  # tune: higher removes larger speckles

        # Spatial filter (edge-preserving smoothing)
        cfg.postProcessing.spatialFilter.enable = False # Determine necessity per environment
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 1

        # Decimation filter (reduces resolution to improve reliability)
        cfg.postProcessing.decimationFilter.decimationFactor = 2
        cfg.postProcessing.decimationFilter.decimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
        cfg.postProcessing.temporalFilter.enable = False

        # IMPORTANT: these values are in mm.
        # set ranges in millimeters.
        cfg.postProcessing.thresholdFilter.minRange = 250    # 0.25 m
        cfg.postProcessing.thresholdFilter.maxRange = 10000   # 10.0 m

        stereo.initialConfig.set(cfg)

        # ---------------------------
        # IMU
        # ---------------------------
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 200)  # 200 Hz
        imu.setBatchReportThreshold(2)

        # ---------------------------
        # OUTPUT (XLink) / Sync
        # ---------------------------
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=1/FPS * 1000 / 2))  # ~ 1/2 frame.
        sync.setSyncAttempts(0) # Set to -1 for default. This should reduce overall latency.
        cam.isp.link(sync.inputs["rgb"]) # Was ISP. Video is faster. ISP may be best for OpenCV processing. 
        stereo.depth.link(sync.inputs["depth"])

        # Create XLink outputs for synced frames.
        xout_sync = pipeline.create(dai.node.XLinkOut)
        xout_sync.setStreamName("synced")
        sync.out.link(xout_sync.input)

        # Create XLink outputs for the IMU stream. This is done separately to reduce overall latency. 
        xout_imu = pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("imu")
        imu.out.link(xout_imu.input)

        return pipeline
    
    def start_pipeline(self):
        """
        Start the DepthAI pipeline and configure the output queues.
        
        :param self: Description
        """
        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()

        # Enable active stereo. Helps in low-light / textureless conditions. TODO: TUNE PER ENVIRONMENT
        self.device.setIrLaserDotProjectorIntensity(1)  # 0: off, 1: on
        self.device.setIrLaserDotProjectorBrightness(250)  # mA, 0..1200

        # Store the image and IMU queues.
        self.synced_queue = self.device.getOutputQueue("synced")
        
        self.synced_queue.setBlocking(False)
        self.synced_queue.setMaxSize(1)

        self.imu_queue = self.device.getOutputQueue("imu")

        self.imu_queue.setBlocking(False)
        self.imu_queue.setMaxSize(5)
        
    def stop_pipeline(self):
        """
        Stop the DepthAI pipeline and release the device.
        """
        if self.device:
            self.device.close()
            self.device = None
            self.pipeline = None
            self.synced_queue = None
            self.imu_queue = None
        
    def get_frames(self):
        """
        Retrieve the latest color frame (BGR format) alongside the latest depth frame. (in mm)
        :return The color frame (BGR)
        :return The depth frame (in mm)
        :retype: Optional[Tuple[np.ndarray, np.ndarray]]
        :rtype: Tuple[np.ndarray, np.ndarray] or None
        """
        # Verify that the pipeline has been started.
        if self.synced_queue is None:
            raise RuntimeError("DepthAI pipeline has not been started.")

        # Attempt to get the latest synced frames. Verify that it exists and is the correct type.
        msg = self.synced_queue.tryGet()
        if msg is None or not isinstance(msg, dai.MessageGroup):
            return None
        
         # 2) Convert group -> dict via iteration (Shown in vendor sample code)
        msgs: Dict[str, dai.ADatatype] = {name: msg for name, msg in msg} 

        rgb_msg = msgs.get("rgb")
        depth_msg = msgs.get("depth")

        if rgb_msg is None or depth_msg is None:
            raise KeyError(f"Missing keys in MessageGroup. Got: {list(msgs.keys())}")

        # 3) Verify inner types
        if not isinstance(rgb_msg, dai.ImgFrame):
            raise TypeError(f'Expected "rgb" to be dai.ImgFrame, got {type(rgb_msg)}')
        if not isinstance(depth_msg, dai.ImgFrame):
            raise TypeError(f'Expected "depth" to be dai.ImgFrame, got {type(depth_msg)}')
        
        return rgb_msg.getCvFrame(), depth_msg.getFrame()  # depth is in mm

    def get_imu_reading(self):
        """
        Retrieves the latest IMU data in the rotation vector format.

        NOTE: This is separate from the synchronized stream to reduce latency and help improve performance.
        :return The rotation vector from the IMU data
        :rtype: 
        """

        # Verify that the pipeline and imu queue have been created.
        if self.device is None or self.imu_queue is None:
            raise RuntimeError("DepthAI pipeline has not been started.")

        # Attempt to get the latest IMU data from the queue.
        imu_msg = self.imu_queue.tryGet()

        # Verify that the IMU datatype is what is expected.
        if not isinstance(imu_msg, dai.IMUData):
            raise TypeError(f'Expected "imu" to be dai.IMUData, got {type(imu_msg)}')
        
        # Verify that IMU data contains packets.
        if len(imu_msg.packets) == 0 or imu_msg.packets[0] is None or not isinstance(imu_msg.packets[0], dai.IMUPacket):
            raise ValueError("IMU data contains no packets.")
        
        # Return the latest IMU data giving heading.
        return imu_msg.packets[0].rotationVector

    def get_intrinsics(self, socket: dai.CameraBoardSocket = dai.CameraBoardSocket.RGB, width: int = 1280, height: int = 800) -> np.ndarray:
        """
        Get the intrinsic parameters of the camera.

        :return: np.ndarray: The intrinsic matrix.
        """
        if self.device is None:
            raise RuntimeError("DepthAI pipeline has not been started.")

        calibration = self.device.readCalibration()
        return np.array(calibration.getCameraIntrinsics(socket, width, height), dtype=np.float32)
    
    def pipeline_active(self):
        """
        Returns whether or not the DepthAI pipeline is currently running.

        :return: 
        """
        if self.device is None:
            return False
        return self.device.isPipelineRunning() and not self.device.isClosed()