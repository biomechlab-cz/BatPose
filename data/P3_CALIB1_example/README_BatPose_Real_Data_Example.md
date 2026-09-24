# BatPose real-data worked example: P3 configuration 1

This package contains one real stereo-camera recording, its matching calibration, and the BatPose outputs exported for that recording. It shows how to take a paired recording through the application and compare new exports with supplied results. It is an example of the software workflow, not a standalone validation against Qualisys.

## Files

| File | Purpose |
| --- | --- |
| `P3_1_left.avi` and `P3_1_right.avi` | Left and right input videos, in that order. Both are 1920 × 1200 pixels, 20 frames/s, 966 frames (48.3 s). |
| `calibration1.yml` | Matching stereo intrinsics, lens model, extrinsics and floor-aligned world frame. Its recorded RMS reprojection error is 1.0324675600314674 pixels, the same value reported in the supplied export metadata. |
| `P3_1.csv` | Exported 3D joint coordinates and confidence (966 frame rows; 17 keypoints). |
| `P3_1.npz` | Exported arrays `joints3d` (966 × 1 × 17 × 3), `conf3d` (966 × 1 × 17), `repro_err` (966 × 1 × 17), and `meta`. |
| `P3_1_angles.csv` | Exported angle series (966 frame rows; eight bilateral joint angles and trunk inclination). |
| `P3_1_metadata.json` | Export metadata, including units, frame rate, coordinate frame, software version and calibration RMS. |
| `P3_1_timestamps.csv` | Exported left and right hardware timestamp fields; interpretation is qualified below. |
| `flash_events_left.csv` | Flash-event annotations from the left video, with source video paths removed. |
| `MANIFEST.json` | Sizes and SHA-256 digests for the supplied files. |

The package contains reconstructed 3D coordinates, confidence, reprojection error and final angles. It does **not** contain separate 2D-keypoint arrays or Qualisys reference data.

## Run the example

1. Install BatPose from the software repository specified with the manuscript. Record the precise release or commit used for any repeat run.
2. In BatPose's existing-video workflow, choose `P3_1_left.avi` as the left video and `P3_1_right.avi` as the right video. Keep this order.
3. Choose `calibration1.yml` as the stereo calibration and click **Run Pipeline**.
4. After processing, manually export the 3D coordinates, angles, NPZ archive, metadata and timestamps. Save these new files outside this package so that the supplied outputs remain intact.
5. Compare the new exports with the supplied outputs. Check frame count and indexing, 20 frames/s, one subject, 17 keypoints, coordinate units (metres), the world coordinate frame, angle columns, confidence and missing-value patterns, and the coordinate and angle time series. Export dates will differ. Exact numerical equivalence and a suitable tolerance have **not** yet been established by an independent repeat run.

The supplied coordinate CSV, angle CSV and timestamp CSV each have 966 rows. The NPZ arrays have 966 frames. Angle column names containing `Flex` denote unsigned geometric angles between keypoint-defined segments, not signed anatomical flexion rotations. The nine-angle landmark definitions and equations are given in the revised manuscript. A finite coordinate with zero confidence should not be interpreted as a fresh valid observation.

## Synchronization and timing

The videos each contain 966 frames at 20 frames/s. The author reports that the visible flash is simultaneous in the two camera views. The left-video annotations use **one-based frame numbers**, with flash-on at frames 46 and 854 and flash-off at frames 81 and 951. Corresponding right-video event frames have not been independently annotated in this package. These observations do not establish sub-frame synchronization or rule out drift.

The hardware timestamp CSV reports non-zero differences between its left and right timestamp fields, approximately 2.07 s in its first row. Their clock origins and export semantics have not been established. Do not interpret that column as a measured video lag or apply an offset from it without examining the capture implementation. Verify the flash and later corresponding movements visually if using the recording for timing-sensitive analysis.

## Scope and release checks

The supplied outputs illustrate the workflow. Exact numerical agreement with a repeat run has not been established.

When repeating the analysis, record the software commit and processing settings, assess agreement with the supplied outputs and verify right-video flash alignment. This package supports inspection and rerunning of the real-data workflow; it does not itself establish clinical measurement validity.

## Download and permission

This example is stored in `data/P3_CALIB1_example` in the BatPose repository. The AVI files use Git LFS. Install Git LFS before cloning and run `git lfs pull` to retrieve the videos; a small text pointer is not the video itself. Verify the downloaded files against `MANIFEST.json`.

The participant has granted permission for public distribution of these recordings.
