# Dataset Conversion Pipeline: nuScenes to KITTI to Xtreme1

This repository contains scripts and tools to convert the nuScenes dataset into KITTI format, and then convert the KITTI dataset into the Xtreme1 format. This pipeline enables interoperability between different dataset formats commonly used in autonomous driving research.

---

## Folder Structure

```File_Name/
├── nuscenes_data/                  # Original nuScenes dataset directory
│   └── ...                        # nuScenes raw data files
├── kitti_output/                  # Output directory for KITTI formatted data
│   ├── CAM_BACK/
│   │   ├── calib/
│   │   ├── image_2/
│   │   └── velodyne/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   └── ...                       # KITTI dataset structure per camera view
├── xtreme1_output/                # Output directory for Xtreme1 formatted data
│   ├── camera_config/
│   └── lidar_point_cloud_0/
├── nuscenes_to_kitti_complete.py # Script to convert nuScenes to KITTI format
├── convert_kitti_to_xtreme1_v2.py # Script to convert KITTI to Xtreme1 format
├── run_conversion.py              # Optional script to run the full pipeline
└── requirements.txt               # Python dependencies for the conversion scripts
```

---

## Scripts Overview

### 1. `nuscenes_to_kitti_complete.py`

- Converts the nuScenes dataset located in `nuscenes_data/` into KITTI format.
- Outputs the KITTI formatted dataset into `kitti_output/`.
- Handles multiple camera views and sensor data.
- Usage example:
  ```bash
  python nuscenes_to_kitti_complete.py --nuscenes_dir nuscenes_data --output_dir kitti_output
  ```

### 2. `convert_kitti_to_xtreme1_v2.py`

- Converts the KITTI dataset located in `kitti_output/` into Xtreme1 format.
- Outputs the Xtreme1 formatted dataset into `xtreme1_output/`.
- Processes camera calibration files, images, and Velodyne point clouds.
- Usage example:
  ```bash
  python convert_kitti_to_xtreme1_v2.py --kitti_dir kitti_output --output_dir xtreme1_output
  ```

---

## Installation and Requirements

Create a Python virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
Pillow
```

---

## Running the Conversion Pipeline

1. **Convert nuScenes to KITTI**

```bash
python nuscenes_to_kitti_complete.py --nuscenes_dir nuscenes_data --output_dir kitti_output
```

2. **Convert KITTI to Xtreme1**

```bash
python convert_kitti_to_xtreme1_v2.py --kitti_dir kitti_output --output_dir xtreme1_output
```

---

## Notes

- Ensure that the input directories (`nuscenes_data` and `kitti_output`) exist and contain the expected dataset files.
- The scripts will print warnings if expected files or directories are missing.
- The Xtreme1 output directory will contain camera configuration JSON files and PCD point cloud files.
- You can customize the Velodyne point cloud directory by using the `--velodyne_dir` argument in `convert_kitti_to_xtreme1_v2.py`.

---

## Contact

For questions or issues, please contact .
