#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI to Xtreme1 Converter

This script converts KITTI format dataset to Xtreme1 format, transforming:
- Camera calibration files to Xtreme1 camera configuration format
- Velodyne point cloud files (.bin) to PCD format

Features:
- Multi-camera support (front, back, left, right views)
- Automatic image dimension detection
- Robust error handling and logging
- Progress tracking during conversion

Author: Prajwal Achar T
Date: May 2, 2025
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar library


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CalibrationParser:
    """Parser for KITTI calibration files."""
    
    @staticmethod
    def parse_kitti_calib(calib_file: str) -> Dict[str, np.ndarray]:
        """
        Parse KITTI calibration file and extract camera parameters.
        
        Args:
            calib_file: Path to the calibration file
            
        Returns:
            Dictionary containing calibration parameters
        """
        data = {}
        try:
            with open(calib_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if not value:
                        logger.warning(f"Empty value for key '{key}' in {calib_file}")
                        continue
                        
                    try:
                        float_values = [float(x) for x in value.split() if x]
                        if float_values:
                            data[key] = np.array(float_values)
                        else:
                            logger.warning(f"No valid float values for key '{key}' in {calib_file}")
                    except ValueError as e:
                        logger.error(f"Failed to parse values for key '{key}' in {calib_file}: {value}")
                        logger.error(f"ValueError: {e}")
                        continue
                        
            if not data:
                logger.error(f"No valid data parsed from {calib_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to read calibration file {calib_file}: {e}")
            return {}


class CameraConfigGenerator:
    """Generator for Xtreme1 camera configuration."""
    
    @staticmethod
    def create_camera_external_transform(calib_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create 4x4 camera_external transform matrix using calibration matrices.
        
        Uses R0_rect and Tr_velo_to_cam from calibration data.
        
        Args:
            calib_data: Dictionary containing calibration parameters
            
        Returns:
            4x4 transformation matrix
        """
        R_rect = np.eye(4)
        if "R0_rect" in calib_data and np.any(calib_data["R0_rect"] != 0):
            R_rect_3x3 = calib_data["R0_rect"].reshape(3, 3)
            R_rect[:3, :3] = R_rect_3x3
        
        Tr_velo_to_cam = np.eye(4)
        if "Tr_velo_to_cam" in calib_data and np.any(calib_data["Tr_velo_to_cam"] != 0):
            Tr = calib_data["Tr_velo_to_cam"].reshape(3, 4)
            Tr_velo_to_cam[:3, :4] = Tr
        
        camera_external = R_rect @ Tr_velo_to_cam
        camera_external = camera_external.T
        return camera_external
    
    @staticmethod
    def create_xtreme1_camera_config(
        calib_data: Dict[str, np.ndarray], 
        img_width: int, 
        img_height: int, 
        cam_idx: int
    ) -> Dict[str, Any]:
        """
        Create camera configuration in Xtreme1 format.
        
        Args:
            calib_data: Dictionary containing calibration parameters
            img_width: Image width in pixels
            img_height: Image height in pixels
            cam_idx: Camera index
            
        Returns:
            Dictionary containing camera configuration
        """
        P_key = f"P{cam_idx}"
        if P_key in calib_data and np.any(calib_data[P_key] != 0):
            P = calib_data[P_key].reshape(3, 4)
        elif "P2" in calib_data and np.any(calib_data["P2"] != 0):
            P = calib_data["P2"].reshape(3, 4)
        else:
            # Default projection matrix if none is available
            P = np.array([
                [1000, 0, img_width/2, 0],
                [0, 1000, img_height/2, 0],
                [0, 0, 1, 0]
            ])
            logger.warning(f"Using default projection matrix for camera {cam_idx}")
            
        # Extract intrinsic parameters
        fx = P[0, 0]
        fy = P[1, 1]
        cx = P[0, 2]
        cy = P[1, 2]
        
        # Create camera external transform
        camera_external = CameraConfigGenerator.create_camera_external_transform(calib_data)
        transform_flat = camera_external.flatten().tolist()
        
        # Format camera configuration
        camera_config = {
            "camera_internal": {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy)
            },
            "width": img_width,
            "height": img_height,
            "camera_external": transform_flat,
            "rowMajor": False
        }
        
        return camera_config


class PointCloudConverter:
    """Converter for Velodyne point cloud files to PCD format."""
    
    @staticmethod
    def convert_velodyne_to_pcd(velodyne_file: str, output_pcd_file: str) -> bool:
        """
        Convert Velodyne binary format (.bin) to PCD format.
        
        Args:
            velodyne_file: Path to the input Velodyne file
            output_pcd_file: Path to the output PCD file
            
        Returns:
            True if conversion was successful, False otherwise
        """
        if not os.path.exists(velodyne_file):
            logger.warning(f"Velodyne file {velodyne_file} not found. Skipping.")
            return False
            
        try:
            with open(velodyne_file, 'rb') as f:
                data = f.read()
                
            # Each point is 4 floats (x, y, z, intensity) with 4 bytes each
            num_points = len(data) // 16
            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
            
            # Write PCD header and data
            with open(output_pcd_file, 'wb') as f:
                f.write(f"# .PCD v0.7 - Point Cloud Data file format\n".encode())
                f.write(f"VERSION 0.7\n".encode())
                f.write(f"FIELDS x y z intensity\n".encode())
                f.write(f"SIZE 4 4 4 4\n".encode())
                f.write(f"TYPE F F F F\n".encode())
                f.write(f"COUNT 1 1 1 1\n".encode())
                f.write(f"WIDTH {num_points}\n".encode())
                f.write(f"HEIGHT 1\n".encode())
                f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n".encode())
                f.write(f"POINTS {num_points}\n".encode())
                f.write(f"DATA binary\n".encode())
                points.tofile(f)
                
            logger.info(f"Converted {velodyne_file} to {output_pcd_file} with {num_points} points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {velodyne_file} to PCD: {e}")
            return False


class KittiToXtreme1Converter:
    """Main converter class for KITTI to Xtreme1 format."""
    
    def __init__(
        self, 
        kitti_dir: str, 
        output_dir: str, 
        velodyne_dir: Optional[str] = None
    ):
        """
        Initialize the converter.
        
        Args:
            kitti_dir: Path to KITTI dataset directory
            output_dir: Path to output Xtreme1 format directory
            velodyne_dir: Path to Velodyne point cloud directory (optional)
        """
        self.kitti_dir = Path(kitti_dir)
        self.output_dir = Path(output_dir)
        self.velodyne_dir = Path(velodyne_dir) if velodyne_dir else self.kitti_dir / "velodyne"
        
        # Define standard camera views
        self.camera_views = ["CAM_BACK", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
        
        # Create output directories
        self.camera_config_dir = self.output_dir / "camera_config"
        self.lidar_point_cloud_dir = self.output_dir / "lidar_point_cloud_0"
        self.camera_config_dir.mkdir(parents=True, exist_ok=True)
        self.lidar_point_cloud_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        if not self.velodyne_dir.exists():
            logger.warning(
                f"Velodyne directory {self.velodyne_dir} not found. "
                "Point cloud conversion will be skipped."
            )
    
    def discover_all_frames(self) -> List[str]:
        """
        Discover all frame numbers available in the dataset.
        
        Returns:
            List of frame numbers (as strings)
        """
        frame_nums = set()
        for camera_view in self.camera_views:
            calib_dir = self.kitti_dir / camera_view / "calib"
            if not calib_dir.exists():
                logger.warning(f"Calibration directory {calib_dir} not found. Skipping.")
                continue
                
            calib_files = [f for f in os.listdir(calib_dir) if f.endswith('.txt')]
            for calib_file in calib_files:
                frame_num = os.path.splitext(calib_file)[0]
                frame_nums.add(frame_num)
                
        return sorted(frame_nums)
    
    def get_image_dimensions(self, camera_view: str, frame_num: str) -> Tuple[int, int]:
        """
        Get image dimensions from image file.
        
        Args:
            camera_view: Camera view name
            frame_num: Frame number
            
        Returns:
            Tuple of (width, height)
        """
        image_dir = self.kitti_dir / camera_view / "image_2"
        img_width, img_height = 1920, 1080  # Default dimensions
        
        # Try reading image size from PNG or JPG file
        for ext in ['.png', '.jpg']:
            image_file = image_dir / f"{frame_num}{ext}"
            if image_file.exists():
                try:
                    with Image.open(image_file) as img:
                        img_width, img_height = img.size
                        break
                except Exception as e:
                    logger.warning(f"Failed to read image size from {image_file}: {e}")
                    
        return img_width, img_height
    
    def process_frame(self, frame_num: str) -> bool:
        """
        Process a single frame.
        
        Args:
            frame_num: Frame number
            
        Returns:
            True if processing was successful, False otherwise
        """
        all_camera_configs = []
        
        # Process each camera view
        for cam_idx, camera_view in enumerate(self.camera_views):
            calib_path = self.kitti_dir / camera_view / "calib" / f"{frame_num}.txt"
            
            if not calib_path.exists():
                logger.warning(
                    f"Calibration file {calib_path} not found. "
                    f"Skipping camera view {camera_view} for frame {frame_num}."
                )
                continue
                
            # Parse calibration data
            calib_data = CalibrationParser.parse_kitti_calib(str(calib_path))
            if not calib_data:
                logger.warning(f"No valid calibration data for {calib_path}. Skipping.")
                continue
                
            # Get image dimensions
            img_width, img_height = self.get_image_dimensions(camera_view, frame_num)
            
            # Create camera config
            camera_config = CameraConfigGenerator.create_xtreme1_camera_config(
                calib_data, img_width, img_height, cam_idx
            )
            all_camera_configs.append(camera_config)
            
        if not all_camera_configs:
            logger.warning(f"No camera configs created for frame {frame_num}. Skipping frame.")
            return False
            
        # Save camera configuration
        config_file = self.camera_config_dir / f"{frame_num}.json"
        with open(config_file, 'w') as f:
            json.dump(all_camera_configs, f, indent=2)
            
        # Convert point cloud if available
        velodyne_file = self.velodyne_dir / f"{frame_num}.bin"
        output_pcd = self.lidar_point_cloud_dir / f"{frame_num}.pcd"
        
        if velodyne_file.exists():
            PointCloudConverter.convert_velodyne_to_pcd(str(velodyne_file), str(output_pcd))
        else:
            logger.warning(
                f"Velodyne file {velodyne_file} not found. "
                f"Skipping point cloud conversion for frame {frame_num}."
            )
            
        return True
    
    def convert(self) -> None:
        """
        Run the full conversion process from KITTI to Xtreme1 format.
        """
        # Discover all frames
        all_frames = self.discover_all_frames()
        logger.info(f"Found {len(all_frames)} unique frames to process.")
        
        # Process each frame with progress bar
        successful_frames = 0
        for frame_num in tqdm(all_frames, desc="Converting frames"):
            if self.process_frame(frame_num):
                successful_frames += 1
                
        # Report results
        logger.info(f"Conversion completed. Output saved to {self.output_dir}")
        logger.info(
            f"Successfully processed {successful_frames}/{len(all_frames)} frames "
            f"with up to {len(self.camera_views)} camera views each."
        )


def main():
    """Parse command line arguments and run the converter."""
    parser = argparse.ArgumentParser(
        description='Convert KITTI format dataset to Xtreme1 format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--kitti_dir', 
        required=True, 
        help='Path to KITTI dataset directory'
    )
    parser.add_argument(
        '--output_dir', 
        required=True, 
        help='Path to output Xtreme1 format directory'
    )
    parser.add_argument(
        '--velodyne_dir', 
        help='Path to Velodyne point cloud directory (optional)'
    )
    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run conversion
    converter = KittiToXtreme1Converter(
        args.kitti_dir, 
        args.output_dir, 
        args.velodyne_dir
    )
    converter.convert()


if __name__ == "__main__":
    main()