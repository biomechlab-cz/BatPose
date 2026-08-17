#!/usr/bin/env bash
# Five independent stereo calibrations, one per board sweep recording.
# Board + lens taken from the GUI calibration that produced
# data/Calibration test/calibration.yml (ChArUco 5x7, 40 mm / 24 mm, fisheye).
set -u
cd "C:/Users/janhe/source/repos/BatPose" || exit 1

CAP="data/Calibration test/capture"
OUT="out/validation"
mkdir -p "$OUT/logs"

i=0
for p in 20260814_111239 20260814_111434 20260814_111636 20260814_111832 20260814_112035; do
  i=$((i + 1))
  echo "=== run$i ($p) start $(date +%H:%M:%S) ==="
  ./.venv/Scripts/python.exe -m app.calib \
    --left  "$CAP/${p}_left.avi" \
    --right "$CAP/${p}_right.avi" \
    --board charuco \
    --squares-x 5 --squares-y 7 \
    --square-size 0.04 --marker-size 0.024 \
    --aruco-dict DICT_6X6_250 \
    --lens fisheye \
    --max-frames 40 --sample-every 5 \
    --out "$OUT/run$i.yml" > "$OUT/logs/run$i.log" 2>&1
  rc=$?
  echo "=== run$i ($p) exit=$rc $(date +%H:%M:%S) ==="
  tail -4 "$OUT/logs/run$i.log"
done
echo "ALL DONE $(date +%H:%M:%S)"
