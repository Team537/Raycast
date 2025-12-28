import depthai as dai
import numpy as np

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
        self.video_queue = None
        self.depth_queue = None 

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
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setSize(1280, 800)
        cam.setPreviewSize(1280, 800)
        cam.setFps(35)
        cam.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

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
        
        mono_left.setFps(35)
        mono_right.setFps(35)

        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT) 
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # 4) Configure and link the cameras to the the depth node.
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setLeftRightCheck(True)

        stereo.setSubpixel(True) # Improve precision at the cost of performance.
        stereo.setSubpixelFractionalBits(4)
        stereo.setExtendedDisparity(False) # Extended disparity increases the range of distances that can be measured.

        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        stereo.enableDistortionCorrection(True) # Enable on-device distortion correction for the stereo pair.

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
        cfg.postProcessing.spatialFilter.enable = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 1

        # Decimation filter (reduces resolution to improve reliability)
        cfg.postProcessing.decimationFilter.decimationFactor = 2
        cfg.postProcessing.decimationFilter.decimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN

        # IMPORTANT: these values are in mm.
        # set ranges in millimeters.
        cfg.postProcessing.thresholdFilter.minRange = 250    # 0.25 m
        cfg.postProcessing.thresholdFilter.maxRange = 10000   # 10.0 m

        stereo.initialConfig.set(cfg)

        # ---------------------------
        # OUTPUT (XLink)
        # ---------------------------
        xout_video = pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam.isp.link(xout_video.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        # Use depth (not disparity) for actual distance work
        stereo.depth.link(xout_depth.input)

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
        self.device.setIrLaserDotProjectorBrightness(200)  # mA, 0..1200
        self.device.setIrFloodLightBrightness(0)           # mA, 0..1500x   

        # Store the image queue.
        self.video_queue = self.device.getOutputQueue("video")
        self.depth_queue = self.device.getOutputQueue("depth")

        self.depth_queue.setBlocking(False)
        self.video_queue.setBlocking(False)
        self.depth_queue.setMaxSize(4)
        self.video_queue.setMaxSize(4)
        
    def stop_pipeline(self):
        """
        Stop the DepthAI pipeline and release the device.
        """
        if self.device:
            self.device.close()
            self.device = None
            self.pipeline = None
            self.video_queue = None
            self.depth_queue = None

    def get_color_frame(self):
        """
        Retrieve the latest color frame. (RGB format)
        """
        # Verify that the pipeline has been started.
        if self.video_queue is None:
            raise RuntimeError("DepthAI pipeline has not been started.")

        # Attempt to get the latest color frame.
        msg = self.video_queue.tryGet()
        if msg is None:
            return None

        # Return the color frame (uint8).
        if isinstance(msg, dai.ImgFrame):
            return msg.getCvFrame()  # uint8 color frame (RGB)
        raise TypeError(f"Video stream returned {type(msg)} (expected ImgFrame)")

    def get_depth_frame_mm(self):
        """
        Retrieve the latest depth frame in millimeters.
        """
        # Verify that the pipeline has been started.
        if self.depth_queue is None:
            raise RuntimeError("DepthAI pipeline has not been started.")

        # Attempt to get the latest depth frame.
        msg = self.depth_queue.tryGet()
        if msg is None:
            return None

        # Return the depth frame in millimeters (uint16).
        if isinstance(msg, dai.ImgFrame):
            return msg.getFrame()  # uint16 depth (mm)
        raise TypeError(f"Depth stream returned {type(msg)} (expected ImgFrame)")