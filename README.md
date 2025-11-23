# README.md

A lightweight, zero-dependency Python tool to convert **MANO** hand pose data into **SMPL-X** compatible `.npz` format.

This tool solves the compatibility gap between hand-only trackers and full-body SMPL-X pipelines by mapping MANO parameters directly to the corresponding SMPL-X hand and global parameters.

## ✨ Features

* **Automatic Hand Detection:** Automatically reads `is_right` flags to assign data to `right_hand_pose` or `left_hand_pose`.
* **Multi-Track Support:** Handles outputs containing single hands, two hands, or multiple motion tracks, generating separate files for each.
* **Direct Mapping:** Exploits the coordinate system compatibility between MANO and SMPL-X (no complex matrix math required).
* **Batch Processing:** Convert entire directories of results in one go.

## 🛠️ Dependencies

* Python 3.x
* NumPy

```bash
pip install numpy
````

## 🚀 Usage

### 1\. Convert a single file

```bash
python convert_to_smplx.py path/to/input_mano.npz
```

### 2\. Specify output path

```bash
python convert_to_smplx.py input.npz -o output_smplx.npz
```

### 3\. Batch convert a directory

```bash
python convert_to_smplx.py path/to/results_dir/ --batch
```

## 🧠 How It Works

SMPL-X is effectively a superset of MANO. This script maps parameters based on the following logic:

1.  **Coordinate System:** MANO and SMPL-X share the same global coordinate definition.
      * MANO `root_orient` $\rightarrow$ SMPL-X `global_orient`
      * MANO `trans` $\rightarrow$ SMPL-X `transl`
2.  **Hand Pose:**
      * If **Right Hand**: Maps MANO `pose_body` (45 dims) $\rightarrow$ SMPL-X `right_hand_pose`.
      * If **Left Hand**: Maps MANO `pose_body` (45 dims) $\rightarrow$ SMPL-X `left_hand_pose`.
3.  **Body Pose:** Since Dyn-HaMR only tracks hands, the SMPL-X `body_pose` (63 dims) is filled with zeros (T-Pose).

## 📂 Output Format

The generated `.npz` file contains the standard SMPL-X parameter dictionary:

| Key | Shape | Description |
| :--- | :--- | :--- |
| `global_orient` | (T, 3) | Global rotation (axis-angle) |
| `transl` | (T, 3) | Global translation |
| `left_hand_pose` | (T, 45) | Left hand joint rotations |
| `right_hand_pose` | (T, 45) | Right hand joint rotations |
| `body_pose` | (T, 63) | Zero-filled (T-pose) |
| `betas` | (10,) | Shape parameters |

## 🧪 Testing

A test script is included to verify the integrity of the conversion:

```bash
python test_conversion.py
```

## 📝 License

[MIT License](https://www.google.com/search?q=LICENSE)

