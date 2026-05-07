# CPU-first 2D Pose Backend Research

**Date:** 2026-02-19
**Author:** Research Agent
**Status:** Final – drives ADR entries in `docs/decisions.md`

---

## 1. Candidates

Three backends were evaluated for CPU-first exercise-video use on a desktop workstation:

| Criterion | MediaPipe Pose Landmarker | MoveNet TFLite | RTMPose (via rtmlib + ONNXRuntime) |
|---|---|---|---|
| **Keypoints** | 33 landmarks (extended body) | 17 (COCO) | 17 (COCO), 133 (wholebody optional) |
| **Multi-person** | No (single-person only) | Up to 6 (MultiPose Lightning) | Yes (unlimited, top-down) |
| **CPU FPS (desktop x86)** | ~27 FPS (Full model, M1 Pro)<br>30+ FPS (Lite, mid-range x86) | Lightning: ~60–90 FPS (192×256)<br>Thunder: ~30–40 FPS (256×256) | RTMPose-s: ~70 FPS (i7-11700, ONNX 1-thread)<br>RTMPose-m: ~50 FPS |
| **Accuracy (single person)** | Good; holistic body incl. feet/hands | Lightning: moderate; Thunder: good | RTMPose-m: 75.8 AP COCO (best in class) |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 (mmpose + rtmlib) |
| **Install complexity** | `pip install mediapipe` – single wheel, self-contained models | `pip install tflite-runtime tensorflow-hub` – model download at runtime | `pip install rtmlib onnxruntime` + ONNX model files (auto-download) |
| **Python API maturity** | Stable since 2022; well-documented | Stable; Google-maintained | rtmlib wrapper is lightweight and actively maintained |
| **Confidence outputs** | Per-landmark visibility + presence scores | Per-keypoint confidence (0–1) | Per-keypoint confidence via detection + pose score |
| **Robustness on video** | Excellent: internal frame-to-frame tracking, re-detects when lost | Good: Lightning re-detects every frame; Thunder more stable | Excellent: top-down design + person detector keeps identity stable |
| **Exercise-video fit** | Excellent: full body, feet, wrist details; handles side-on and back-facing poses | Good: fast, but fewer landmarks; may miss feet accuracy | Excellent: highest accuracy, better occlusion handling |
| **GPU upgrade path** | GPU delegate (optional) | GPU delegate (optional) | TensorRT / CUDA ONNX provider |
| **Dependency footprint** | ~35 MB wheel | ~3 MB tflite-runtime + ~10 MB model | ~20 MB rtmlib + ~15–30 MB ONNX models |

---

## 2. Detailed Notes per Backend

### 2.1 MediaPipe Pose Landmarker (Google)

- **Architecture:** BlazePose – two-stage (detector + landmark regressor). Runs detector once, then tracks with a lightweight model.
- **Output:** 33 3D landmarks (x, y, z_relative, visibility, presence). Only x, y used for 2D pose; z is a weak depth estimate.
- **Single-person limitation:** The standard Pose Landmarker API is designed for one prominent person. A workaround is to run per-detected-bounding-box, but there is no built-in multi-person pipeline.
- **CPU real-time:** Achieves 27–30 FPS with the Full model on mid-range x86 CPU via XNNPACK delegate. Lite model is faster with slightly lower accuracy.
- **Strengths for this project:** Full-body coverage (shoulders → feet, wrists, heels), excellent tracking stability for exercise scenarios (stationary or slow-moving subject), zero additional dependency after install.
- **Weakness:** Single-person only per API call; 33-landmark topology requires mapping to canonical COCO-17 for downstream code.

### 2.2 MoveNet (Google/TensorFlow)

- **Architecture:** Single-stage, heatmap-based. Lightning (fast) and Thunder (accurate) variants.
- **Output:** 17 COCO keypoints, confidence per joint.
- **Multi-person:** MultiPose Lightning handles up to 6 people with bounding-box tracking. However, MultiPose currently runs CPU-only (no GPU delegate).
- **CPU real-time:** Lightning runs at 60–90 FPS on desktop; Thunder at ~30–40 FPS. Very fast.
- **Accuracy:** Lower than RTMPose on COCO benchmarks. In comparative studies, MoveNet Lightning had lower PDJ than MediaPipe Full for key biomechanical joints (hips, knees).
- **Weakness:** Fewer landmarks than MediaPipe (no wrist/hand detail, no heel landmarks). Less robust for side-facing or back-facing exercise poses. TensorFlow Hub model download adds friction.

### 2.3 RTMPose (OpenMMLab) via `rtmlib`

- **Architecture:** Top-down (person detector → per-person pose regressor via SimCC). Requires a separate person detector ONNX model.
- **Install:** `pip install rtmlib onnxruntime` (no mmcv/mmdet/mmpose required via rtmlib wrapper). ONNX models are auto-downloaded on first run.
- **Output:** 17 COCO keypoints (or 133 wholebody). Confidence per joint.
- **Multi-person:** Full multi-person support. Number of tracked people limited only by detector speed.
- **CPU real-time:** RTMPose-s: ~70 FPS; RTMPose-m: ~50 FPS on Intel i7-11700 with ONNXRuntime single-thread. This includes the person detector step.
- **Accuracy:** Best in class at 75.8 AP on COCO (RTMPose-m), significantly outperforming MoveNet Thunder.
- **Weakness:** Two-model pipeline (detector + pose); adds ~100–300 ms latency to first frame; ONNX model files (~30–50 MB) must be downloaded. Slightly more setup friction than MediaPipe.

---

## 3. Recommendation

### 3.1 Default Backend (POC and MVP): **MediaPipe Pose Landmarker (Full model)**

**Rationale:**
- Zero-friction install (`pip install mediapipe`), no external file downloads at runtime.
- 33 landmarks provide richer body coverage than COCO-17 alone—particularly wrists and heels, which matter for biomechanics/exercise analysis.
- 27–30 FPS on a CPU workstation meets the real-time requirement for 30 Hz camera input.
- Excellent tracking stability: the internal detector-tracker loop reduces jitter without manual smoothing work.
- Best fit for the primary use case: **one or two subjects** in an exercise session with mostly frontal or sagittal views.
- Apache 2.0: no commercial restrictions.

**Known limitations and mitigations:**
- Single-person per API call → run one inference per detected bounding box if multi-person needed (acceptable for 2-person max exercise scenario).
- 33→17 COCO mapping needed → documented in `docs/skeleton_mapping.md`.

### 3.2 Optional Upgrade Backend: **RTMPose-m via rtmlib**

**Rationale:**
- Best CPU accuracy (75.8 AP COCO). Use when the exercise session has multiple people or when precise joint localization is needed (e.g., subtle knee-angle analysis).
- Outputs native COCO-17 keypoints → no landmark mapping required.
- Already runs at 50 FPS on a mid-range CPU; GPU upgrade is straightforward (CUDA ONNX provider).
- Same NPZ output interface means zero changes to downstream triangulation and smoothing code.

**Upgrade path:** Implemented behind a `--backend rtmpose` flag in `python -m app.pose2d`. The implementation agent should design the `PoseBackend` interface so that both backends share the same `detect(frame) → (keypoints_NxJx2, conf_NxJ)` signature.

---

## 4. Speed vs Accuracy Summary

```
Accuracy (AP/PDJ) HIGH
     |          RTMPose-m ●
     |        RTMPose-s ●
     |    MediaPipe Full ●
     |  MoveNet Thunder ●
     |  MoveNet Lightning ●
     +--------------------------------> CPU Speed (FPS) HIGH
                    MoveNet Lit  RTMPose-s  MediaPipe   RTMPose-m
                    ~80 FPS      ~70 FPS    ~27 FPS     ~50 FPS
```

For the CPU-only workstation use case, **MediaPipe Full** sits in a sweet spot: adequate FPS for 30 Hz video with the best landmark coverage for the domain.

---

## 5. References

- [MediaPipe BlazePose blog post](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/)
- [MoveNet TensorFlow Hub tutorial](https://www.tensorflow.org/hub/tutorials/movenet)
- [RTMPose paper (arXiv 2303.07399)](https://arxiv.org/abs/2303.07399)
- [rtmlib GitHub (lightweight RTMPose wrapper)](https://github.com/Tau-J/rtmlib)
- [Best Pose Estimation Models – Roboflow blog](https://blog.roboflow.com/best-pose-estimation-models/)
- [Comparative Analysis: OpenPose/PoseNet/MoveNet (ResearchGate)](https://www.researchgate.net/publication/359476644_Comparative_Analysis_of_OpenPose_PoseNet_and_MoveNet_Models_for_Pose_Estimation_in_Mobile_Devices)
