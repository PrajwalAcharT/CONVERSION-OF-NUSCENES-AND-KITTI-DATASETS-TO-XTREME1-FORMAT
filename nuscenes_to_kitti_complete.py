#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NuScenes to KITTI Format Converter

This module provides functionality to convert data from the NuScenes
dataset format to the KITTI dataset format. It supports conversion
of lidar points, camera images, calibration information, and 3D
bounding box annotations.

Author: Prajwal Achar T
Date: May 2, 2025
"""

import os
import sys
import json
import time
import uuid
import shutil
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import traceback

import numpy as np
import cv2
import fire
from tqdm import tqdm
from joblib import Parallel, delayed
from pyquaternion import Quaternion

# NuScenes imports
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from nuscenes.utils.kitti import KittiDB


# ===== Configuration =====

@dataclass
class Config:
    """
    Configuration settings for NuScenes to KITTI conversion.
    
    This class centralizes all configuration parameters and provides
    validation and sensible defaults.
    """
    # Processing settings
    num_processes: int = field(default_factory=lambda: min(os.cpu_count() or 4, 8))
    retry_max_attempts: int = 3
    retry_delay: int = 1
    
    # Cache settings
    lidar_cache_size: int = 10
    
    # Image settings
    imsize: Tuple[int, int] = (1600, 900)
    
    # Camera settings
    cam_names: List[str] = field(default_factory=lambda: ["CAM_FRONT"])
    
    # Dataset settings
    image_count: Optional[int] = 10
    
    # Annotation settings
    difficulty_level: int = 0
    min_bbox_area: float = 25.0
    
    # Visualization settings
    visualize: bool = False
    
    # Output directory structure
    output_folders: List[str] = field(default_factory=lambda: ["label_2", "calib", "image_2", "velodyne"])
    
    # Additional settings
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_processes < 1:
            self.num_processes = 1
            logging.warning("Number of processes must be at least 1, setting to 1")
        
        if self.retry_max_attempts < 1:
            self.retry_max_attempts = 1
            logging.warning("Retry attempts must be at least 1, setting to 1")
        
        if self.min_bbox_area < 0:
            self.min_bbox_area = 0
            logging.warning("Minimum bounding box area must be non-negative, setting to 0")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            Config: New configuration object with values from the file
        """
        if not os.path.exists(config_path):
            logging.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert keys to lowercase
            config_dict = {k.lower(): v for k, v in config_dict.items()}
            
            # Create instance with default values
            config = cls()
            
            # Update with values from file
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logging.warning(f"Unknown configuration parameter: {key}")
            
            return config
        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {e}")
            return cls()


# ===== Exceptions =====

class NuScenesConverterError(Exception):
    """Base exception for all converter errors."""
    pass


class DatasetInitializationError(NuScenesConverterError):
    """Exception raised when the NuScenes dataset cannot be initialized."""
    pass


class InvalidAnnotationError(NuScenesConverterError):
    """Exception raised when an annotation is invalid or cannot be converted."""
    pass


class FileOperationError(NuScenesConverterError):
    """Exception raised when a file operation fails."""
    pass


class ConversionError(NuScenesConverterError):
    """Exception raised when conversion of a sample fails."""
    pass


# ===== Utility Functions =====

def setup_logging(level: int = logging.INFO) -> str:
    """
    Set up logging with a specific format and level.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        str: Session ID for this conversion run
    """
    session_id = str(uuid.uuid4())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(session_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add session_id to all log records
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.session_id = session_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Conversion session started with ID: {session_id}")
    
    return session_id


def retry(func):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        func: Function to be retried
        
    Returns:
        Function wrapper that implements retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get config from the first argument if it's a class method
        # or from the global config if it's a standalone function
        if args and hasattr(args[0], 'config'):
            config = args[0].config
        else:
            config = Config()
        
        logger = logging.getLogger(func.__module__)
        last_exception = None
        
        for attempt in range(config.retry_max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.retry_max_attempts} "
                    f"failed for {func.__name__}: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
        
        logger.error(f"Function {func.__name__} failed after {config.retry_max_attempts} attempts")
        if config.debug:
            logger.error(f"Last exception: {traceback.format_exc()}")
        
        raise last_exception or RuntimeError(f"Function {func.__name__} failed repeatedly")
    
    return wrapper


def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    
    Raises:
        FileOperationError: If directory creation fails
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise FileOperationError(f"Failed to create directory {path}: {e}")


def ensure_3d_numpy(array: Any) -> np.ndarray:
    """
    Ensure input is a 3D numpy array, adding singleton dimension if 2D.
    
    Args:
        array: Input array-like object
        
    Returns:
        np.ndarray: 3D numpy array
        
    Raises:
        ValueError: If array is not 2D or 3D
    """
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    elif array.ndim != 3:
        raise ValueError("Input must be a 2D or 3D array")
    return array


def is_visible(box: Box, intrinsic: np.ndarray, imsize: Tuple[int, int]) -> bool:
    """
    Check if a 3D box is visible in the image plane.
    
    Args:
        box: 3D bounding box
        intrinsic: Camera intrinsic matrix
        imsize: Image dimensions (width, height)
        
    Returns:
        bool: True if the box is visible, False otherwise
    """
    try:
        # Get corners of the 3D box
        corners = box.corners()
        
        # Ensure correct format
        corners = ensure_3d_numpy(corners).reshape(3, -1)
        
        # Create projection matrix
        proj_mat = np.hstack((intrinsic, np.zeros((3, 1))))
        
        # Add homogeneous coordinate
        corners_hom = np.vstack((corners, np.ones(corners.shape[1])))
        
        # Project points to image plane
        pts = proj_mat @ corners_hom
        
        # Check if any points are behind the camera
        if np.any(pts[2] <= 0):
            return False
        
        # Normalize projected points
        pts_norm = pts[:2] / pts[2:]
        
        # Check if any points are within image bounds
        x_visible = np.logical_and(0 <= pts_norm[0], pts_norm[0] < imsize[0])
        y_visible = np.logical_and(0 <= pts_norm[1], pts_norm[1] < imsize[1])
        
        return np.any(np.logical_and(x_visible, y_visible))
    except Exception as e:
        logging.warning(f"Error checking visibility: {e}")
        return False


# ===== Converter Components =====

def create_calibration_file(calib_path: str, cs_record: Dict[str, Any]) -> None:
    """
    Write KITTI-style calibration file.
    
    Args:
        calib_path: Path to save the calibration file
        cs_record: Calibrated sensor record
        
    Raises:
        FileOperationError: If file writing fails
    """
    try:
        # Extract camera intrinsic matrix
        intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # Create projection matrix (3x4)
        P = np.column_stack([intrinsic, np.zeros(3)])
        
        with open(calib_path, "w") as calib_file:
            # Write projection matrices for all cameras (only first is actual)
            for i in range(4):
                if i == 0:
                    # Actual camera matrix
                    matrix_str = " ".join(map(str, P.flatten()))
                else:
                    # Placeholder matrices
                    matrix_str = "0 " * 12
                
                calib_file.write(f'P{i}: {matrix_str}\n')
    except Exception as e:
        raise FileOperationError(f"Error writing calibration file: {e}")


def convert_box_to_kitti_format(
    annotation: Dict[str, Any],
    velo_to_cam_kitti: np.ndarray,
    cs_record: Dict[str, Any],
    config: Config
) -> Optional[Tuple[Box, np.ndarray]]:
    """
    Convert NuScenes 3D bounding box annotation to KITTI format.
    
    Args:
        annotation: NuScenes annotation dictionary
        velo_to_cam_kitti: Transformation matrix from velodyne to camera
        cs_record: Calibrated sensor record
        config: Configuration object
        
    Returns:
        Tuple containing KITTI box and 2D bounding box, or None if conversion fails
        
    Raises:
        InvalidAnnotationError: If annotation doesn't contain required fields
    """
    required_fields = ['translation', 'rotation', 'size', 'category_name']
    if not all(field in annotation for field in required_fields):
        raise InvalidAnnotationError(f"Annotation missing required fields: {required_fields}")
    
    try:
        # Extract box parameters
        translation = np.array(annotation['translation'])
        rotation = Quaternion(annotation['rotation'])
        size = np.array(annotation['size'])
        
        # Create NuScenes box
        box = Box(translation, size, rotation)
        
        # Transform to sensor coordinate frame
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        # Convert to KITTI format
        kitti_box = KittiDB.box_nuscenes_to_kitti(
            box=box,
            velo_to_cam_rot=Quaternion(matrix=velo_to_cam_kitti[:3, :3]),
            velo_to_cam_trans=velo_to_cam_kitti[:3, 3],
            r0_rect=Quaternion()
        )
        
        # Project 3D box to 2D
        bbox_2d = KittiDB.project_kitti_box_to_image(
            box=kitti_box,
            intrinsic=np.array(cs_record['camera_intrinsic']),
            imsize=tuple(config.imsize)
        )
        
        # Check if box is visible and meets minimum size
        if bbox_2d is not None:
            bbox_width = bbox_2d[2] - bbox_2d[0]
            bbox_height = bbox_2d[3] - bbox_2d[1]
            
            if bbox_width * bbox_height < config.min_bbox_area:
                logging.debug(f"Bounding box too small: {bbox_width * bbox_height} < {config.min_bbox_area}")
                return None
            
            return kitti_box, bbox_2d
        else:
            return None
    except Exception as e:
        logging.debug(f"Box conversion failed: {e}")
        return None


def create_label_file(
    label_path: str,
    annotations: List[Dict[str, Any]],
    velo_to_cam_kitti: np.ndarray,
    cs_record: Dict[str, Any],
    config: Config
) -> None:
    """
    Write KITTI-style label file with 3D bounding boxes.
    
    Args:
        label_path: Path to save the label file
        annotations: List of NuScenes annotations
        velo_to_cam_kitti: Transformation matrix from velodyne to camera
        cs_record: Calibrated sensor record
        config: Configuration object
        
    Raises:
        FileOperationError: If file writing fails
    """
    try:
        with open(label_path, "w") as label_file:
            valid_annotations = 0
            
            for annotation in annotations:
                try:
                    conversion_result = convert_box_to_kitti_format(
                        annotation, velo_to_cam_kitti, cs_record, config
                    )
                    
                    if conversion_result is None:
                        continue
                    
                    kitti_box, bbox_2d = conversion_result
                    
                    # Create KITTI label format string
                    label_str = KittiDB.box_to_string(
                        name=annotation['category_name'],
                        box=kitti_box,
                        bbox_2d=bbox_2d,
                        truncation=0.0,
                        occlusion=config.difficulty_level
                    )
                    
                    label_file.write(label_str + '\n')
                    valid_annotations += 1
                except Exception as e:
                    logging.warning(f"Failed to create label for annotation: {e}")
                    continue
            
            logging.debug(f"Wrote {valid_annotations} annotations to {label_path}")
    except Exception as e:
        raise FileOperationError(f"Error writing label file: {e}")


# ===== Main Converter Class =====

class KittiConverter:
    """
    Converter from NuScenes dataset format to KITTI format.
    
    This class handles the conversion of LiDAR point clouds, camera images,
    calibration information, and 3D bounding box annotations from NuScenes
    to KITTI format.
    """
    
    def __init__(
        self,
        nusc_kitti_dir: str = '~/kitti_output',
        config_path: Optional[str] = None,
        cam_names: Optional[List[str]] = None,
        lidar_name: str = 'LIDAR_TOP',
        image_count: Optional[int] = None,
        nusc_version: str = 'v1.0-mini',
        split: str = 'mini_train',
        dataroot: Optional[str] = None,
        visualize: bool = False,
        debug: bool = False
    ):
        """
        Initialize the NuScenes to KITTI converter.
        
        Args:
            nusc_kitti_dir: Directory to store KITTI format data
            config_path: Path to configuration file
            cam_names: List of camera names to process
            lidar_name: Name of the LiDAR sensor
            image_count: Number of images to convert
            nusc_version: NuScenes dataset version
            split: Dataset split
            dataroot: Path to NuScenes dataset
            visualize: Whether to generate visualization images
            debug: Enable debug mode
        """
        # Initialize configuration
        self.config = Config.from_file(config_path) if config_path else Config()
        
        # Override config with constructor parameters
        if cam_names:
            self.config.cam_names = cam_names
        if image_count:
            self.config.image_count = image_count
        self.config.visualize = visualize
        self.config.debug = debug
        
        # Set logging level based on debug flag
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Setup paths and parameters
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.lidar_name = lidar_name
        self.nusc_version = nusc_version
        self.split = split
        self.dataroot = dataroot
        
        # LiDAR data cache
        self.lidar_cache = {}
        
        # Initialize NuScenes dataset
        self.nusc = self._initialize_nuscenes()
        
        # Create output directories
        self._setup_output_directories()
        
        # Log initial configuration
        logging.info(f"Initialized converter with {self.nusc_version} dataset, "
                     f"output to {self.nusc_kitti_dir}, "
                     f"cameras: {self.config.cam_names}")
    
    def _setup_output_directories(self) -> None:
        """
        Create output directory structure for KITTI format data.
        """
        base_dir = os.path.join(self.nusc_kitti_dir, self.split)
        
        # Create standard KITTI folders
        for folder in self.config.output_folders:
            ensure_directory(os.path.join(base_dir, folder))
        
        # Create visualization folder if needed
        if self.config.visualize:
            ensure_directory(os.path.join(base_dir, 'image_2_overlay'))
    
    @retry
    def _initialize_nuscenes(self) -> NuScenes:
        """
        Initialize NuScenes dataset with robust path handling.
        
        Returns:
            NuScenes: Initialized dataset object
            
        Raises:
            DatasetInitializationError: If dataset initialization fails
        """
        logging.info(f"Initializing NuScenes with version {self.nusc_version}")
        
        # Try multiple possible paths for the dataset
        try_paths = [
            self.dataroot,
            os.path.join(self.dataroot, self.nusc_version) if self.dataroot else None,
            os.path.join(self.dataroot, 'v1.0-mini') if self.dataroot else None,
            os.path.expanduser('~/nuscenes'),
            os.path.expanduser('~/data/nuscenes')
        ]
        
        # Filter out None values
        try_paths = [p for p in try_paths if p is not None]
        
        # Try each path until successful
        for path in try_paths:
            try:
                logging.debug(f"Trying to initialize NuScenes with path: {path}")
                nusc = NuScenes(version=self.nusc_version, dataroot=path, verbose=False)
                logging.info(f"Successfully initialized NuScenes with path: {path}")
                return nusc
            except Exception as e:
                logging.debug(f"Failed with path {path}: {e}")
                continue
        
        raise DatasetInitializationError(
            f"Could not initialize NuScenes dataset. "
            f"Tried paths: {try_paths}. "
            f"Please ensure the dataset is downloaded and the path is correct."
        )
    
    def nuscenes_gt_to_kitti(self, camera_names: Optional[List[str]] = None) -> None:
        """
        Convert NuScenes ground truth to KITTI format.
        
        Args:
            camera_names: List of camera names to process (overrides config)
            
        Raises:
            ConversionError: If conversion process fails
        """
        camera_names = camera_names or self.config.cam_names
        
        # Define transform from KITTI to NuScenes lidar coordinate system
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        
        # Get sample tokens for the specified split
        try:
            logging.info(f"Fetching samples for split: {self.split}")
            sample_tokens = self._get_all_samples_in_split(self.split)
            
            # Limit number of samples if specified
            if self.config.image_count:
                sample_tokens = sample_tokens[:self.config.image_count]
            
            logging.info(f"Converting {len(sample_tokens)} samples to KITTI format...")
            
            # Process samples in parallel
            results = Parallel(
                n_jobs=self.config.num_processes,
                backend='threading',
                verbose=5
            )(
                delayed(self._process_sample)(
                    sample_token, camera_names, kitti_to_nu_lidar, kitti_to_nu_lidar_inv
                ) for sample_token in tqdm(sample_tokens, desc="Converting samples")
            )
            
            # Check for failed samples
            failed_samples = [token for token, success in zip(sample_tokens, results) if not success]
            if failed_samples:
                logging.warning(f"Failed to convert {len(failed_samples)} samples: {failed_samples[:5]}")
            
            # Count converted images
            image_dir = os.path.join(self.nusc_kitti_dir, self.split, 'image_2')
            if os.path.exists(image_dir):
                num_images = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
                logging.info(f"Total images converted: {num_images}")
            else:
                logging.warning("Image directory not found. Conversion might have failed.")
            
            logging.info("Conversion completed successfully.")
            
        except Exception as e:
            logging.error(f"Conversion failed: {e}")
            if self.config.debug:
                logging.error(traceback.format_exc())
            raise ConversionError(f"Failed to convert NuScenes to KITTI: {e}")
    
    @retry
    def _process_sample(
        self,
        sample_token: str,
        camera_names: List[str],
        kitti_to_nu_lidar: Quaternion,
        kitti_to_nu_lidar_inv: Quaternion
    ) -> bool:
        """
        Process a single NuScenes sample and convert to KITTI format.
        
        Args:
            sample_token: Sample token
            camera_names: List of camera names to process
            kitti_to_nu_lidar: Quaternion for KITTI to NuScenes transformation
            kitti_to_nu_lidar_inv: Inverse quaternion for transformation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get sample data
            sample = self.nusc.get('sample', sample_token)
            
            # Get LiDAR data
            lidar_token = sample['data'][self.lidar_name]
            pcl = self._get_lidar_data(lidar_token, kitti_to_nu_lidar_inv)
            
            # Get annotations
            annotations = [
                self.nusc.get('sample_annotation', ann_token)
                for ann_token in sample['anns']
            ]
            
            # Get camera data
            cam_token = sample['data'][camera_names[0]]
            sd_record = self.nusc.get('sample_data', cam_token)
            cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            
            # Calculate transformation matrices
            lid_to_ego = transform_matrix(
                cs_record['translation'],
                Quaternion(cs_record['rotation']),
                inverse=False
            )
            
            ego_to_cam = transform_matrix(
                sd_record['translation'],
                Quaternion(sd_record['rotation']),
                inverse=True
            )
            
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)
            
            # Define output paths
            base_path = self.split
            label_path = os.path.join(self.nusc_kitti_dir, base_path, 'label_2', f'{sample_token}.txt')
            calib_path = os.path.join(self.nusc_kitti_dir, base_path, 'calib', f'{sample_token}.txt')
            image_path = os.path.join(self.nusc_kitti_dir, base_path, 'image_2', f'{sample_token}.png')
            lidar_save_path = os.path.join(self.nusc_kitti_dir, base_path, 'velodyne', f'{sample_token}.bin')
            
            # Copy image file
            try:
                shutil.copy(os.path.join(self.nusc.dataroot, sd_record['filename']), image_path)
            except Exception as e:
                logging.error(f"Failed to copy image file: {e}")
                return False
            
            # Save LiDAR data
            try:
                with open(lidar_save_path, "wb") as f:
                    pcl.points.T.tofile(f)
            except Exception as e:
                logging.error(f"Failed to save LiDAR data: {e}")
                return False
            
            # Create calibration file
            try:
                create_calibration_file(calib_path, cs_record)
            except Exception as e:
                logging.error(f"Failed to create calibration file: {e}")
                return False
            
            # Create label file
            try:
                create_label_file(label_path, annotations, velo_to_cam_kitti, cs_record, self.config)
            except Exception as e:
                logging.error(f"Failed to create label file: {e}")
                return False
            
            # Create visualization if requested
            if self.config.visualize:
                try:
                    self._visualize_sample(image_path, annotations, cs_record, velo_to_cam_kitti)
                except Exception as e:
                    logging.warning(f"Visualization failed: {e}")
                    # Continue even if visualization fails
            
            return True
        
        except Exception as e:
            logging.error(f"Error processing sample {sample_token}: {e}")
            if self.config.debug:
                logging.error(traceback.format_exc())
            return False
    
    def _get_lidar_data(self, lidar_token: str, rotation: Quaternion) -> LidarPointCloud:
        """
        Load LiDAR data with caching.
        
        Args:
            lidar_token: LiDAR data token
            rotation: Rotation to apply to points
            
        Returns:
            LidarPointCloud: Loaded and transformed point cloud
            
        Raises:
            FileOperationError: If LiDAR data cannot be loaded
        """
        lidar_path = self.nusc.get_sample_data_path(lidar_token)
        
        # Check cache first
        if lidar_path in self.lidar_cache:
            return self.lidar_cache[lidar_path]
        
        try:
            # Load LiDAR data
            pcl = LidarPointCloud.from_file(lidar_path)
            
            # Apply rotation
            pcl.rotate(rotation.rotation_matrix)
            
            # Manage cache size
            if len(self.lidar_cache) >= self.config.lidar_cache_size:
                # Remove oldest entry (first key)
                self.lidar_cache.pop(next(iter(self.lidar_cache)))
            
            # Add to cache
            self.lidar_cache[lidar_path] = pcl
            
            return pcl
        except Exception as e:
            raise FileOperationError(f"Failed to load LiDAR data from {lidar_path}: {e}")
    
    def _visualize_sample(
        self,
        image_path: str,
        annotations: List[Dict[str, Any]],
        cs_record: Dict[str, Any],
        velo_to_cam_kitti: np.ndarray
    ) -> None:
        """
        Visualize a sample with 3D bounding boxes projected to the image.
        
        Args:
            image_path: Path to the image file
            annotations: List of annotations
            cs_record: Calibrated sensor record
            velo_to_cam_kitti: Transformation matrix from velodyne to camera
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not read image at {image_path}")
            
            # Draw each annotation on the image
            for annotation in annotations:
                try:
                    # Convert box to KITTI format
                    conversion_result = convert_box_to_kitti_format(
                        annotation, velo_to_cam_kitti, cs_record, self.config
                    )
                    
                    if conversion_result is None:
                        continue
                    
                    kitti_box, bbox_2d = conversion_result
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox_2d)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw category name
                    cv2.putText(
                        img,
                        annotation['category_name'],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
                except Exception as e:
                    logging.debug(f"Failed to visualize annotation: {e}")
                    continue
            
            # Save visualization
            output_path = image_path.replace('image_2', 'image_2_overlay')
            ensure_directory(os.path.dirname(output_path))
            cv2.imwrite(output_path, img)
            logging.debug(f"Visualized image saved to {output_path}")
        except Exception as e:
            logging.warning(f"Error visualizing sample: {e}")
            if self.config.debug:
                logging.warning(traceback.format_exc())