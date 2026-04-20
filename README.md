# Floor Video Map Builder

## Project Identity

This repository contains a downward-facing vehicle video mapping system prepared for **紫荆书院**, **Tsinghua University**, for **Challenge Cup** project presentation and engineering demonstration.

Author: **Lennart A. Conrad**

The core program, `floor_video_map.py`, converts a floor-facing or road-facing video stream into a 2D stitched map. Unlike a phone panorama, it does not assume motion is horizontal. It estimates per-frame motion in map space, accumulates 2D transforms, computes global bounds from transformed frame corners, and renders a map that can grow in any direction.

## What The Project Does

Given a video from a camera mounted near the floor of a vehicle, the pipeline:

1. Samples frames from the input video.
2. Detects and matches image features between frames.
3. Estimates frame-to-frame transforms with RANSAC.
4. Accumulates each accepted frame into a global world coordinate system.
5. Computes canvas bounds from all transformed frame corners.
6. Renders a clean final mosaic plus optional debugging and progress outputs.

This makes it suitable for:

- floor or road texture reconstruction
- vehicle-path visualization
- thermal-floor inspection experiments
- engineering demos where trajectory and map growth must remain readable

## Main Features

- 2D map-style stitching instead of a single panorama strip
- supports straight motion, turns, curves, and in-place rotation
- dynamic canvas growth in all directions
- multiple transform models: translation, partial affine, affine, homography
- frame acceptance based on translation and rotation thresholds
- multiple blending modes, including `best` for faithful single-observation output
- optional progress video, comparison video, and edge-first comparison video
- pose logging to CSV and JSON
- optional tiled rendering for very large outputs
- fallback tracking support with optical flow and ECC refinement

## Repository Structure

```text
floor_video_map.py    main CLI program
requirements.txt      Python dependencies
README.md             project overview and usage
video_runs/           local experiment outputs (ignored by git)
```

## Environment Setup

Recommended Python version: `3.10+`

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or install manually:

```bash
python -m pip install opencv-python numpy tqdm
```

## Quick Start

Use an explicit video path:

```bash
python floor_video_map.py input.mp4 --output mosaic.png
```

If no input path is given, the program auto-detects the first common video file in the working directory:

```bash
python floor_video_map.py --output mosaic.png
```

## Recommended Commands

General RGB or floor-texture mapping:

```bash
python floor_video_map.py input.mp4 \
  --output mosaic.png \
  --output-video progress.mp4 \
  --comparison-video comparison.mp4 \
  --comparison-edge-video comparison_edges.mp4 \
  --video-view overview \
  --every 1 \
  --model partial-affine \
  --detector sift \
  --blend best \
  --preview-blend best \
  --draw-trajectory \
  --draw-current-frame-outline \
  --draw-frame-index \
  --draw-quality \
  --write-poses poses.csv \
  --write-json poses.json \
  --preview preview.png \
  --trajectory trajectory.png
```

Thermal or low-texture data:

```bash
python floor_video_map.py input.avi \
  --output mosaic.png \
  --output-video progress.mp4 \
  --comparison-video comparison.mp4 \
  --comparison-edge-video comparison_edges.mp4 \
  --every 1 \
  --model partial-affine \
  --detector sift \
  --min-matches 8 \
  --min-inliers 6 \
  --min-inlier-ratio 0.2 \
  --blend best \
  --preview-blend best \
  --use-ecc \
  --write-poses poses.csv \
  --write-json poses.json \
  --preview preview.png \
  --trajectory trajectory.png
```

If the canvas becomes too large:

```bash
python floor_video_map.py input.mp4 \
  --output mosaic.png \
  --enable-tiling \
  --tile-size 2048 \
  --tile-output-dir tiles \
  --preview preview.png
```

## Important Output Files

Depending on arguments, the program can generate:

```text
mosaic.png               final clean mosaic
progress.mp4             map-building progress video
comparison.mp4           original / mosaic / trajectory side-by-side video
comparison_edges.mp4     edge-growth comparison video with final mosaic hold
poses.csv                pose log for engineering inspection
poses.json               structured pose log
trajectory.png           top-down path visualization
preview.png              downsampled preview image
tiles/manifest.json      tiled-render metadata
```

## Key CLI Options

- `input`: input video path, optional when auto-detect is used
- `--output`: final mosaic image path
- `--output-video`: progress video path
- `--comparison-video`: side-by-side original, mosaic, and trajectory video
- `--comparison-edge-video`: edge-growth comparison video ending with the final painted mosaic
- `--video-view overview|follow|canvas`: progress-video view mode
- `--start-frame`, `--end-frame`: restrict processing to a frame range
- `--every`: sample every Nth frame
- `--max-frames`: stop after N sampled frames
- `--crop x,y,w,h`: crop the frame before processing
- `--max-dim`: resize the input frame for speed
- `--detector sift|orb`: feature extractor
- `--model translation|partial-affine|affine|homography`: motion model
- `--blend last|average|feather|max|smart|best`: overlap blending strategy
- `--preview-blend inherit|last|average|feather|max|smart|best`: preview-specific blend mode
- `--min-step-px`: minimum world-space translation before accepting a frame
- `--min-rotation-deg`: minimum rotation before accepting a frame
- `--accept-every-frame`: disable motion-based frame filtering
- `--use-ecc`: enable ECC-based alignment refinement
- `--recent-keyframes`: number of recent keyframes used for re-acquisition
- `--motion-history`: number of recent successful motions used for prediction
- `--max-bridge-frames`: how many predicted bridge frames may be used before re-lock
- `--write-poses`: write pose CSV
- `--write-json`: write pose JSON
- `--trajectory`: write trajectory image
- `--preview`: write preview image
- `--enable-tiling`: render output as tiles when the canvas is too large

Run `python floor_video_map.py --help` for the full interface.

## Blending Modes

- `last`: newest frame overwrites old content
- `average`: overlap averaging
- `feather`: soft overlap weighting
- `max`: useful for thermal-style hot-spot preservation
- `smart`: mixed strategy that feathers flat areas and preserves detail
- `best`: picks one strong real observation per output pixel using detail, center, and tracking-quality weighting

For Challenge Cup presentation and engineering readability, `best` is the recommended default because it preserves object boundaries better than averaging.

## Technical Notes

- All motion is represented as `3x3` homogeneous matrices.
- Global map size is computed from transformed frame corners, not from a preallocated strip.
- The pipeline is two-stage: pose estimation first, rendering second.
- The progress and comparison videos are designed to make frame placement and trajectory easy to inspect.
- Tracking fallback logic uses recent keyframes, motion priors, optical flow, and optional ECC.
- Tiled output is supported for large canvases.

## Current Engineering Status

What works well:

- map growth through turns and non-horizontal motion
- readable progress videos for debugging and presentation
- faithful mosaic rendering with the `best` blend mode
- side-by-side comparison outputs for engineering inspection

Current limitations:

- trajectory quality is still bounded by texture quality, motion blur, and thermal contrast
- low-texture or blank floor regions may require stronger cropping, denser sampling, or ECC fallback
- this version does not yet include global bundle adjustment or loop-closure optimization

## Suggested Future Work

- global pose-graph refinement across accepted keyframes
- loop-closure detection for long trajectories
- better thermal-camera live capture tooling
- automated crop suggestion
- quantitative trajectory evaluation against external references

## Author And Copyright

- Author: **Lennart A. Conrad**
- Copyright: **Copyright (c) 2026 Lennart A. Conrad**
- License: [MIT](LICENSE)

## Third-Party Components

- This repository's original project code is attributed to **Lennart A. Conrad**.
- Third-party libraries such as `opencv-python`, `numpy`, and `tqdm` remain under their own licenses.
- Input videos, camera outputs, datasets, and external hardware documentation are not relicensed by this repository unless explicitly stated.

## Attribution

Prepared for **紫荆书院**, **Tsinghua University**, for **Challenge Cup** project work and demonstration by **Lennart A. Conrad**.
