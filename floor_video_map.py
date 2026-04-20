#!/usr/bin/env python3
"""Build a floor/road mosaic from a downward-facing vehicle video.

The script estimates a 2D pose for sampled video frames, accumulates those
poses in a shared world/map coordinate system, computes global bounds from the
transformed frame corners, and then renders either a single mosaic image or a
tiled output. An optional progress video shows how the map grows over time.

Author: Lennart A. Conrad
Copyright (c) 2026 Lennart A. Conrad
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Iterable, Sequence

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")
DEFAULT_OUTPUT = "mosaic.png"
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_CODEC = "mp4v"

__author__ = "Lennart A. Conrad"
__copyright__ = "Copyright (c) 2026 Lennart A. Conrad"


@dataclass
class FramePose:
    frame_index: int
    timestamp_sec: float
    image_path_or_index: str | int
    H_frame_to_world: np.ndarray | None
    center_world: tuple[float, float] | None
    corners_world: np.ndarray | None
    num_keypoints: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    accepted: bool
    reason: str


@dataclass
class ProcessedFrame:
    frame_index: int
    timestamp_sec: float
    color: np.ndarray
    gray: np.ndarray
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: np.ndarray | None
    H_frame_to_world: np.ndarray
    tracking_mode: str = "visual"


@dataclass
class TransformEstimate:
    H_current_to_reference: np.ndarray | None
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    reason: str
    matched_current_points: np.ndarray | None = None
    matched_reference_points: np.ndarray | None = None
    used_ecc: bool = False
    used_flow: bool = False
    ecc_score: float | None = None


@dataclass
class RenderResult:
    mosaic_bgr: np.ndarray | None
    valid_mask: np.ndarray | None
    crop_box: tuple[int, int, int, int] | None


@dataclass
class FrameContentAnalysis:
    frame_index: int
    sharpness: float
    detail_mean: float
    tracking_confidence: float
    global_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch a downward-facing floor video into a 2D map mosaic."
    )
    parser.add_argument("input", nargs="?", help="Input video path. Auto-detected if omitted.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Final mosaic image path.")
    parser.add_argument("--output-video", help="Optional progress/output video path.")
    parser.add_argument(
        "--comparison-video",
        help="Optional side-by-side video showing source frame, mosaic build, and trajectory.",
    )
    parser.add_argument(
        "--comparison-edge-video",
        help=(
            "Optional side-by-side video showing source frame, edge-only map development, "
            "then a final painted mosaic hold."
        ),
    )
    parser.add_argument(
        "--edge-final-hold-sec",
        type=float,
        default=1.5,
        help="How long to hold the final painted mosaic in --comparison-edge-video.",
    )
    parser.add_argument(
        "--video-view",
        choices=("overview", "follow", "canvas"),
        default="overview",
        help="Video viewport mode.",
    )
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--video-fps", type=float, default=0.0, help="Defaults to input fps.")
    parser.add_argument("--follow-scale", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to consider.")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Last frame index to consider. Negative values mean until the end.",
    )
    parser.add_argument("--every", type=int, default=2, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N sampled frames.")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1000,
        help="Resize processed frames so the largest dimension is at most this value.",
    )
    parser.add_argument("--crop", help="Crop processed frames as x,y,w,h before resizing.")
    parser.add_argument("--detector", choices=("sift", "orb"), default="sift")
    parser.add_argument("--nfeatures", type=int, default=4000)
    parser.add_argument(
        "--model",
        choices=("translation", "partial-affine", "affine", "homography"),
        default="affine",
    )
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold.")
    parser.add_argument(
        "--ransac",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold in pixels.",
    )
    parser.add_argument("--min-matches", type=int, default=18)
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.3)
    parser.add_argument("--min-step-px", type=float, default=50.0)
    parser.add_argument("--min-rotation-deg", type=float, default=4.0)
    parser.add_argument("--accept-every-frame", action="store_true")
    parser.add_argument(
        "--blend",
        choices=("last", "average", "feather", "max", "smart", "best"),
        default="feather",
    )
    parser.add_argument("--max-canvas-mp", type=float, default=250.0)
    parser.add_argument("--enable-tiling", action="store_true")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--tile-output-dir", default="tiles")
    parser.add_argument("--margin-px", type=int, default=64)
    parser.add_argument("--presentation-align-motion-right", action="store_true")
    parser.add_argument("--draw-trajectory", action="store_true")
    parser.add_argument("--draw-current-frame-outline", action="store_true")
    parser.add_argument("--draw-frame-index", action="store_true")
    parser.add_argument("--draw-quality", action="store_true")
    parser.add_argument("--debug-overlay-on-mosaic", action="store_true")
    parser.add_argument("--no-crop-output", action="store_true")
    parser.add_argument("--write-poses", help="CSV pose log output path.")
    parser.add_argument("--write-json", help="JSON pose log output path.")
    parser.add_argument("--trajectory", help="Optional trajectory debug image path.")
    parser.add_argument("--preview", help="Optional downsampled preview image path.")
    parser.add_argument(
        "--preview-blend",
        choices=("inherit", "last", "average", "feather", "max", "smart", "best"),
        default="inherit",
        help="Blend mode for the preview image. Defaults to the main --blend mode.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-ecc", action="store_true", help="Refine transforms with ECC.")
    parser.add_argument(
        "--disable-flow",
        action="store_true",
        help="Disable optical-flow fallback when descriptor matching is weak.",
    )
    parser.add_argument("--flow-max-points", type=int, default=300)
    parser.add_argument("--flow-min-distance", type=float, default=12.0)
    parser.add_argument("--flow-quality", type=float, default=0.01)
    parser.add_argument("--flow-error-threshold", type=float, default=20.0)
    parser.add_argument("--flow-backcheck-threshold", type=float, default=1.5)
    parser.add_argument("--min-ecc", type=float, default=0.9)
    parser.add_argument("--max-translation-factor", type=float, default=1.5)
    parser.add_argument("--max-rotation-jump-deg", type=float, default=75.0)
    parser.add_argument("--min-scale", type=float, default=0.6)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--min-determinant", type=float, default=0.2)
    parser.add_argument("--max-determinant", type=float, default=5.0)
    parser.add_argument("--bootstrap-min-keypoints", type=int, default=8)
    parser.add_argument("--bootstrap-min-matches", type=int, default=12)
    parser.add_argument("--bootstrap-min-inliers", type=int, default=8)
    parser.add_argument(
        "--allow-ecc-bootstrap",
        action="store_true",
        help="Allow ECC-only initialization when feature-backed bootstrap is unavailable.",
    )
    parser.add_argument("--recent-keyframes", type=int, default=4)
    parser.add_argument("--motion-history", type=int, default=4)
    parser.add_argument("--max-bridge-frames", type=int, default=8)
    return parser.parse_args()


def log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def parse_crop(crop_text: str | None) -> tuple[int, int, int, int] | None:
    if not crop_text:
        return None
    parts = [int(p.strip()) for p in crop_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("--crop width/height must be positive")
    return x, y, w, h


def open_video_or_autodetect(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        print(f"Selected input video: {path}")
        return path

    candidates = sorted(
        path.resolve()
        for path in Path.cwd().iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(
            "No input path provided and no supported video files found in the working directory."
        )
    print(f"No input specified; selected first video file: {candidates[0]}")
    return candidates[0]


def open_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_video_info(video_path: Path) -> dict[str, float | int]:
    cap = open_capture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def resolve_frame_range(total_frames: int, args: argparse.Namespace) -> tuple[int, int]:
    if total_frames <= 0:
        return 0, -1
    start_frame = max(0, int(args.start_frame))
    end_frame = total_frames - 1 if int(args.end_frame) < 0 else min(int(args.end_frame), total_frames - 1)
    if start_frame > end_frame:
        raise ValueError(
            f"Invalid frame range: start {start_frame} is after end {end_frame}. "
            "Adjust --start-frame/--end-frame."
        )
    return start_frame, end_frame


def preprocess_frame(
    frame_bgr: np.ndarray,
    crop: tuple[int, int, int, int] | None,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply crop, resize, grayscale conversion, and contrast normalization."""

    processed = frame_bgr
    if crop:
        x, y, w, h = crop
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + w, frame_bgr.shape[1])
        y1 = min(y + h, frame_bgr.shape[0])
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Crop rectangle falls outside the input frame.")
        processed = processed[y0:y1, x0:x1]

    height, width = processed.shape[:2]
    if max_dim > 0:
        scale = min(1.0, float(max_dim) / float(max(height, width)))
        if scale < 1.0:
            resized_w = max(1, int(round(width * scale)))
            resized_h = max(1, int(round(height * scale)))
            processed = cv2.resize(processed, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return processed, gray


def make_detector(detector_name: str, nfeatures: int):
    if detector_name == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=nfeatures), "sift"
        print("Warning: SIFT is unavailable in this OpenCV build; falling back to ORB.")
        detector_name = "orb"
    if detector_name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures), "orb"
    raise ValueError(f"Unsupported detector: {detector_name}")


def detect_and_describe(
    gray: np.ndarray,
    detector,
) -> tuple[Sequence[cv2.KeyPoint], np.ndarray | None]:
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints or [], descriptors


def make_matcher(detector_name: str):
    if detector_name == "orb":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def match_features(
    matcher,
    current_descriptors: np.ndarray | None,
    reference_descriptors: np.ndarray | None,
    ratio_threshold: float,
) -> list[cv2.DMatch]:
    if current_descriptors is None or reference_descriptors is None:
        return []
    if len(current_descriptors) < 2 or len(reference_descriptors) < 2:
        return []
    raw_matches = matcher.knnMatch(current_descriptors, reference_descriptors, k=2)
    good_matches: list[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if second.distance <= 0:
            continue
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)
    return good_matches


def estimate_rotation_deg_from_H(H: np.ndarray) -> float:
    angle_rad = math.atan2(H[1, 0], H[0, 0])
    return math.degrees(angle_rad)


def normalize_angle_deg(angle_deg: float) -> float:
    value = (angle_deg + 180.0) % 360.0 - 180.0
    if value == -180.0:
        return 180.0
    return value


def estimate_scale_from_H(H: np.ndarray) -> float:
    a = H[:2, :2]
    sx = float(np.linalg.norm(a[:, 0]))
    sy = float(np.linalg.norm(a[:, 1]))
    if not np.isfinite(sx) or not np.isfinite(sy):
        return float("nan")
    return math.sqrt(max(1e-12, sx * sy))


def matrix_to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    if matrix.shape == (2, 3):
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = matrix
        return H
    raise ValueError(f"Unsupported transform shape: {matrix.shape}")


def refine_transform_with_ecc(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    model: str,
    H_init: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    if model == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp = H_init[:2, :].astype(np.float32)
    elif model in ("partial-affine", "affine"):
        warp_mode = cv2.MOTION_AFFINE
        warp = H_init[:2, :].astype(np.float32)
    elif model == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp = H_init.astype(np.float32)
    else:
        return None, None

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-6,
    )
    try:
        cc, warp = cv2.findTransformECC(
            reference_gray,
            current_gray,
            warp,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None, None
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return matrix_to_homogeneous(warp), float(cc)
    return matrix_to_homogeneous(warp), float(cc)


def estimate_translation_transform(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(current_points) < 2:
        return None, None
    deltas = reference_points - current_points
    translation = np.median(deltas, axis=0)
    residual = np.linalg.norm(deltas - translation, axis=1)
    inliers = (residual <= ransac_threshold).astype(np.uint8).reshape(-1, 1)
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = float(translation[0])
    H[1, 2] = float(translation[1])
    return H, inliers


def estimate_transform_from_correspondences(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(current_points) != len(reference_points):
        raise ValueError("Point correspondence arrays must have the same length.")
    if len(current_points) == 0:
        return None, None

    if args.model == "translation":
        return estimate_translation_transform(current_points, reference_points, args.ransac)
    if args.model == "partial-affine":
        matrix, inliers = cv2.estimateAffinePartial2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac,
            maxIters=5000,
            confidence=0.995,
            refineIters=10,
        )
        return (None if matrix is None else matrix_to_homogeneous(matrix), inliers)
    if args.model == "affine":
        matrix, inliers = cv2.estimateAffine2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac,
            maxIters=5000,
            confidence=0.995,
            refineIters=10,
        )
        return (None if matrix is None else matrix_to_homogeneous(matrix), inliers)
    if args.model == "homography":
        H, inliers = cv2.findHomography(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac,
            maxIters=5000,
            confidence=0.995,
        )
        return (None if H is None else matrix_to_homogeneous(H), inliers)
    raise ValueError(f"Unsupported model: {args.model}")


def collect_flow_points(
    reference_frame: ProcessedFrame,
    args: argparse.Namespace,
) -> np.ndarray | None:
    points: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    for keypoint in sorted(reference_frame.keypoints, key=lambda kp: kp.response, reverse=True):
        pt = keypoint.pt
        key = (int(round(pt[0])), int(round(pt[1])))
        if key in seen:
            continue
        seen.add(key)
        points.append((float(pt[0]), float(pt[1])))
        if len(points) >= args.flow_max_points:
            break

    remaining = args.flow_max_points - len(points)
    if remaining > 0:
        extra = cv2.goodFeaturesToTrack(
            reference_frame.gray,
            maxCorners=remaining,
            qualityLevel=args.flow_quality,
            minDistance=args.flow_min_distance,
            blockSize=7,
        )
        if extra is not None:
            for pt in extra.reshape(-1, 2):
                key = (int(round(float(pt[0]))), int(round(float(pt[1]))))
                if key in seen:
                    continue
                seen.add(key)
                points.append((float(pt[0]), float(pt[1])))
                if len(points) >= args.flow_max_points:
                    break

    if len(points) < 4:
        return None
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def estimate_flow_transform(
    reference_frame: ProcessedFrame,
    current_gray: np.ndarray,
    args: argparse.Namespace,
) -> TransformEstimate:
    reference_points = collect_flow_points(reference_frame, args)
    if reference_points is None:
        return TransformEstimate(
            H_current_to_reference=None,
            num_matches=0,
            num_inliers=0,
            inlier_ratio=0.0,
            reason="optical flow fallback found too few corners",
            used_flow=True,
        )

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        reference_frame.gray,
        current_gray,
        reference_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return TransformEstimate(
            H_current_to_reference=None,
            num_matches=0,
            num_inliers=0,
            inlier_ratio=0.0,
            reason="optical flow tracking failed",
            used_flow=True,
        )

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        reference_frame.gray,
        next_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    status = status.reshape(-1).astype(bool)
    if back_points is None or back_status is None:
        backcheck_mask = np.zeros_like(status, dtype=bool)
    else:
        back_status = back_status.reshape(-1).astype(bool)
        fb_error = np.linalg.norm(
            back_points.reshape(-1, 2) - reference_points.reshape(-1, 2),
            axis=1,
        )
        backcheck_mask = back_status & (fb_error <= args.flow_backcheck_threshold)
    if error is None:
        error_mask = np.ones_like(status, dtype=bool)
    else:
        error_mask = error.reshape(-1) <= args.flow_error_threshold
    valid = status & error_mask & backcheck_mask
    if np.count_nonzero(valid) < 4:
        return TransformEstimate(
            H_current_to_reference=None,
            num_matches=int(np.count_nonzero(valid)),
            num_inliers=0,
            inlier_ratio=0.0,
            reason="optical flow fallback found too few tracked points",
            used_flow=True,
        )

    tracked_current_points = next_points.reshape(-1, 2)[valid].astype(np.float32)
    tracked_reference_points = reference_points.reshape(-1, 2)[valid].astype(np.float32)
    H, inliers = estimate_transform_from_correspondences(
        tracked_current_points,
        tracked_reference_points,
        args,
    )
    if H is None or inliers is None:
        return TransformEstimate(
            H_current_to_reference=None,
            num_matches=len(tracked_current_points),
            num_inliers=0,
            inlier_ratio=0.0,
            reason="optical flow transform estimation failed",
            matched_current_points=tracked_current_points,
            matched_reference_points=tracked_reference_points,
            used_flow=True,
        )

    num_matches = len(tracked_current_points)
    num_inliers = int(np.count_nonzero(inliers))
    inlier_ratio = float(num_inliers / max(1, num_matches))
    return TransformEstimate(
        H_current_to_reference=H,
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        reason="ok",
        matched_current_points=tracked_current_points,
        matched_reference_points=tracked_reference_points,
        used_flow=True,
    )


def estimate_pair_transform(
    reference_frame: ProcessedFrame,
    current_gray: np.ndarray,
    current_keypoints: Sequence[cv2.KeyPoint],
    current_descriptors: np.ndarray | None,
    matcher,
    args: argparse.Namespace,
    H_init_current_to_reference: np.ndarray | None = None,
) -> TransformEstimate:
    def ecc_fallback(reason_prefix: str) -> TransformEstimate:
        H_init = H_init_current_to_reference
        if H_init is None:
            H_init = np.eye(3, dtype=np.float64)
        H_ecc, ecc_score = refine_transform_with_ecc(
            reference_frame.gray,
            current_gray,
            args.model,
            H_init,
        )
        if H_ecc is None:
            return TransformEstimate(
                H_current_to_reference=None,
                num_matches=num_matches,
                num_inliers=0,
                inlier_ratio=0.0,
                reason=f"{reason_prefix}; ECC fallback failed",
            )
        return TransformEstimate(
            H_current_to_reference=H_ecc,
            num_matches=num_matches,
            num_inliers=0,
            inlier_ratio=0.0,
            reason=f"{reason_prefix}; ECC fallback",
            used_ecc=True,
            ecc_score=ecc_score,
        )

    def flow_fallback(reason_prefix: str) -> TransformEstimate:
        flow_estimate = estimate_flow_transform(reference_frame, current_gray, args)
        if flow_estimate.H_current_to_reference is None:
            if args.use_ecc:
                return ecc_fallback(f"{reason_prefix}; {flow_estimate.reason}")
            flow_estimate.reason = f"{reason_prefix}; {flow_estimate.reason}"
            return flow_estimate
        if args.use_ecc:
            H_ecc, ecc_score = refine_transform_with_ecc(
                reference_frame.gray,
                current_gray,
                args.model,
                flow_estimate.H_current_to_reference,
            )
            if H_ecc is not None:
                flow_estimate.H_current_to_reference = H_ecc
                flow_estimate.ecc_score = ecc_score
        flow_estimate.reason = f"{reason_prefix}; optical flow fallback"
        return flow_estimate

    matches = match_features(
        matcher,
        current_descriptors,
        reference_frame.descriptors,
        args.ratio,
    )
    num_matches = len(matches)
    if num_matches < args.min_matches:
        if not args.disable_flow:
            return flow_fallback(f"too few matches ({num_matches} < {args.min_matches})")
        if args.use_ecc:
            return ecc_fallback(f"too few matches ({num_matches} < {args.min_matches})")
        return TransformEstimate(
            H_current_to_reference=None,
            num_matches=num_matches,
            num_inliers=0,
            inlier_ratio=0.0,
            reason=f"too few matches ({num_matches} < {args.min_matches})",
        )

    current_points = np.float32([current_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    reference_points = np.float32(
        [reference_frame.keypoints[m.trainIdx].pt for m in matches]
    ).reshape(-1, 2)

    H, inliers = estimate_transform_from_correspondences(
        current_points,
        reference_points,
        args,
    )

    if H is None or inliers is None:
        if not args.disable_flow:
            return flow_fallback("transform estimation failed")
        if args.use_ecc:
            return ecc_fallback("transform estimation failed")
        return TransformEstimate(
            H_current_to_reference=None,
            num_matches=num_matches,
            num_inliers=0,
            inlier_ratio=0.0,
            reason="transform estimation failed",
        )

    if args.use_ecc:
        H_ecc, ecc_score = refine_transform_with_ecc(
            reference_frame.gray,
            current_gray,
            args.model,
            H,
        )
        if H_ecc is not None:
            H = H_ecc
    else:
        ecc_score = None

    num_inliers = int(np.count_nonzero(inliers))
    inlier_ratio = float(num_inliers / max(1, num_matches))
    if num_inliers < args.min_inliers or inlier_ratio < args.min_inlier_ratio:
        if not args.disable_flow:
            flow_estimate = flow_fallback(
                f"weak feature transform ({num_inliers} inliers, ratio {inlier_ratio:.2f})"
            )
            if flow_estimate.H_current_to_reference is not None:
                return flow_estimate
    if args.use_ecc and (num_inliers < args.min_inliers or inlier_ratio < args.min_inlier_ratio):
        return ecc_fallback(
            f"weak feature transform ({num_inliers} inliers, ratio {inlier_ratio:.2f})"
        )
    return TransformEstimate(
        H_current_to_reference=H,
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        reason="ok",
        matched_current_points=current_points,
        matched_reference_points=reference_points,
        used_ecc=False,
        ecc_score=ecc_score,
    )


def validate_transform(
    H_current_to_reference: np.ndarray,
    estimate: TransformEstimate,
    frame_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if H_current_to_reference is None:
        return False, estimate.reason
    if not np.isfinite(H_current_to_reference).all():
        return False, "transform contains NaN or Inf"
    if estimate.used_ecc:
        if estimate.ecc_score is None:
            return False, "ECC did not report a correlation score"
        if estimate.ecc_score < args.min_ecc:
            return False, f"ECC score {estimate.ecc_score:.3f} below {args.min_ecc:.3f}"
    else:
        if estimate.num_matches < args.min_matches:
            return False, f"too few matches ({estimate.num_matches})"
        if estimate.num_inliers < args.min_inliers:
            return False, f"too few inliers ({estimate.num_inliers})"
        if estimate.inlier_ratio < args.min_inlier_ratio:
            return False, f"inlier ratio too low ({estimate.inlier_ratio:.2f})"

    affine = H_current_to_reference[:2, :2]
    det = float(np.linalg.det(affine))
    if not np.isfinite(det):
        return False, "affine determinant is invalid"
    if det < args.min_determinant or det > args.max_determinant:
        return False, f"determinant {det:.3f} outside [{args.min_determinant}, {args.max_determinant}]"

    scale = estimate_scale_from_H(H_current_to_reference)
    if not np.isfinite(scale):
        return False, "scale estimate is invalid"
    if scale < args.min_scale or scale > args.max_scale:
        return False, f"scale {scale:.3f} outside [{args.min_scale}, {args.max_scale}]"

    rotation_deg = abs(normalize_angle_deg(estimate_rotation_deg_from_H(H_current_to_reference)))
    if rotation_deg > args.max_rotation_jump_deg:
        return False, f"rotation jump {rotation_deg:.2f} deg exceeds {args.max_rotation_jump_deg}"

    frame_h, frame_w = frame_shape[:2]
    frame_diag = math.hypot(frame_w, frame_h)
    translation = float(np.linalg.norm(H_current_to_reference[:2, 2]))
    max_translation = args.max_translation_factor * frame_diag
    if translation > max_translation:
        return False, f"translation jump {translation:.1f}px exceeds {max_translation:.1f}px"

    return True, "ok"


def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H.astype(np.float64))
    return transformed.reshape(-1, 2)


def compute_frame_geometry(H_frame_to_world: np.ndarray, width: int, height: int) -> tuple[tuple[float, float], np.ndarray]:
    corners = np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    corners_world = transform_points(H_frame_to_world, corners)
    center = transform_points(
        H_frame_to_world,
        np.array([[width * 0.5, height * 0.5]], dtype=np.float64),
    )[0]
    return (float(center[0]), float(center[1])), corners_world


def should_accept_pose(
    pose_center_world: tuple[float, float],
    pose_rotation_deg: float,
    last_accepted_pose: FramePose,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if args.accept_every_frame:
        return True, "accept-every-frame"
    if last_accepted_pose.center_world is None or last_accepted_pose.H_frame_to_world is None:
        return True, "first usable frame"

    dx = pose_center_world[0] - last_accepted_pose.center_world[0]
    dy = pose_center_world[1] - last_accepted_pose.center_world[1]
    step = math.hypot(dx, dy)
    last_rotation_deg = estimate_rotation_deg_from_H(last_accepted_pose.H_frame_to_world)
    rotation_delta = abs(normalize_angle_deg(pose_rotation_deg - last_rotation_deg))
    if step >= args.min_step_px:
        return True, f"step {step:.1f}px >= {args.min_step_px:.1f}px"
    if rotation_delta >= args.min_rotation_deg:
        return True, f"rotation {rotation_delta:.1f}deg >= {args.min_rotation_deg:.1f}deg"
    return False, (
        f"below thresholds (step={step:.1f}px, rotation={rotation_delta:.1f}deg)"
    )


def estimate_transform_score(
    estimate: TransformEstimate,
    source_name: str,
    current_frame_index: int,
    reference_frame_index: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Rank valid estimates for trajectory stability.

    The scorer prefers stronger geometry, but it also biases toward temporally
    closer references so the tracker does not snap back to older keyframes
    unless they are clearly more reliable.
    """

    frame_gap = max(1, current_frame_index - reference_frame_index)
    if estimate.used_ecc:
        method_rank = 0.0
        quality = estimate.ecc_score or -1.0
    elif estimate.used_flow:
        method_rank = 1.0
        quality = estimate.inlier_ratio
    else:
        method_rank = 2.0
        quality = estimate.inlier_ratio

    if source_name == "previous":
        source_rank = 2.0
    elif source_name == "accepted":
        source_rank = 1.0
    else:
        source_rank = 0.0

    temporal_confidence = float(estimate.num_inliers) / float(frame_gap)
    return (
        method_rank,
        temporal_confidence,
        quality,
        float(estimate.num_inliers),
        source_rank,
        -float(frame_gap),
        float(estimate.num_matches),
    )


def bootstrap_transform_is_reliable(
    estimate: TransformEstimate,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """Keep bootstrap conservative so blank scenes do not become the map anchor."""

    if estimate.used_ecc:
        if not args.allow_ecc_bootstrap:
            return (
                False,
                "bootstrap requires a feature-backed transform; rerun with --allow-ecc-bootstrap to override",
            )
        if estimate.ecc_score is None:
            return False, "bootstrap ECC did not report a correlation score"
        if estimate.ecc_score < args.min_ecc:
            return False, f"bootstrap ECC score {estimate.ecc_score:.3f} below {args.min_ecc:.3f}"
        return True, "ok"

    if estimate.num_matches < args.bootstrap_min_matches:
        return (
            False,
            f"bootstrap matches {estimate.num_matches} below {args.bootstrap_min_matches}",
        )
    if estimate.num_inliers < args.bootstrap_min_inliers:
        return (
            False,
            f"bootstrap inliers {estimate.num_inliers} below {args.bootstrap_min_inliers}",
        )
    return True, "ok"


def append_history_unique(history: Deque[ProcessedFrame], frame: ProcessedFrame) -> None:
    if history and history[-1].frame_index == frame.frame_index:
        history[-1] = frame
        return
    history.append(frame)


def build_reference_candidates(
    last_accepted_frame: ProcessedFrame | None,
    accepted_frame_history: Deque[ProcessedFrame],
    successful_history: Deque[ProcessedFrame],
    args: argparse.Namespace,
) -> list[tuple[str, ProcessedFrame]]:
    candidates: list[tuple[str, ProcessedFrame]] = []
    seen: set[int] = set()

    def add_candidate(label: str, frame: ProcessedFrame | None) -> None:
        if frame is None or frame.frame_index in seen:
            return
        seen.add(frame.frame_index)
        candidates.append((label, frame))

    add_candidate("accepted", last_accepted_frame)

    extra_successful = list(successful_history)[:-1][-args.recent_keyframes :]
    extra_successful.reverse()
    for frame in extra_successful:
        if frame.tracking_mode == "predicted":
            continue
        add_candidate("recent-successful", frame)

    extra_accepted = list(accepted_frame_history)[-args.recent_keyframes :]
    extra_accepted.reverse()
    for frame in extra_accepted:
        add_candidate("recent-accepted", frame)
    return candidates


def make_rotation_matrix_2x2(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


def predict_frame_to_world(
    successful_history: Sequence[ProcessedFrame],
    current_frame_index: int,
    frame_size: tuple[int, int],
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Predict the next world pose from recent successful poses.

    The prediction uses recent center velocity and heading-rate estimates.
    It is deliberately conservative and is mainly used as an alignment prior
    or short-gap bridge, not as a long unbounded dead-reckoning source.
    """

    usable = [frame for frame in successful_history if frame.H_frame_to_world is not None]
    if len(usable) < 2:
        return None
    history = usable[-max(2, args.motion_history) :]
    width, height = frame_size

    centers: list[np.ndarray] = []
    rotations: list[float] = []
    for frame in history:
        center_world, _ = compute_frame_geometry(frame.H_frame_to_world, width, height)
        centers.append(np.array(center_world, dtype=np.float64))
        rotations.append(estimate_rotation_deg_from_H(frame.H_frame_to_world))

    velocity_terms: list[np.ndarray] = []
    angular_terms: list[float] = []
    weights: list[float] = []
    for idx in range(1, len(history)):
        frame_gap = history[idx].frame_index - history[idx - 1].frame_index
        if frame_gap <= 0:
            continue
        velocity_terms.append((centers[idx] - centers[idx - 1]) / float(frame_gap))
        angular_terms.append(
            normalize_angle_deg(rotations[idx] - rotations[idx - 1]) / float(frame_gap)
        )
        weights.append(float(idx))

    if not velocity_terms:
        return None

    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = weights_np / np.sum(weights_np)
    velocity = np.sum(np.stack(velocity_terms, axis=0) * weights_np[:, None], axis=0)
    angular_velocity = float(np.sum(np.asarray(angular_terms, dtype=np.float64) * weights_np))

    last_frame = history[-1]
    frame_delta = current_frame_index - last_frame.frame_index
    if frame_delta <= 0:
        return None

    predicted_center = centers[-1] + velocity * float(frame_delta)
    predicted_rotation = rotations[-1] + angular_velocity * float(frame_delta)
    center_local = np.array([width * 0.5, height * 0.5], dtype=np.float64)

    H_last = last_frame.H_frame_to_world
    if args.model == "translation":
        A_pred = np.eye(2, dtype=np.float64)
    else:
        scale = estimate_scale_from_H(H_last)
        A_pred = make_rotation_matrix_2x2(predicted_rotation) * scale

    translation = predicted_center - (A_pred @ center_local)
    H_pred = np.eye(3, dtype=np.float64)
    H_pred[:2, :2] = A_pred
    H_pred[:2, 2] = translation
    return H_pred


def compose_global_poses(
    video_path: Path,
    crop: tuple[int, int, int, int] | None,
    args: argparse.Namespace,
) -> tuple[list[FramePose], list[FramePose], tuple[int, int]]:
    """Pass 1: estimate frame poses in a shared world coordinate system."""

    video_info = get_video_info(video_path)
    fps = float(video_info["fps"]) if video_info["fps"] else 0.0
    total_frames = int(video_info["frame_count"])
    start_frame, end_frame = resolve_frame_range(total_frames, args)

    detector, detector_name = make_detector(args.detector, args.nfeatures)
    matcher = make_matcher(detector_name)

    cap = open_capture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    all_poses: list[FramePose] = []
    accepted_poses: list[FramePose] = []
    previous_successful_frame: ProcessedFrame | None = None
    last_accepted_frame: ProcessedFrame | None = None
    last_accepted_pose: FramePose | None = None
    bootstrap_candidate_frame: ProcessedFrame | None = None
    bootstrap_candidate_pose_index: int | None = None
    successful_history: Deque[ProcessedFrame] = deque(
        maxlen=max(args.motion_history + args.max_bridge_frames + 4, 16)
    )
    accepted_frame_history: Deque[ProcessedFrame] = deque(
        maxlen=max(args.recent_keyframes + 2, 8)
    )
    consecutive_bridge_frames = 0
    sampled_frames = 0
    processed_shape: tuple[int, int] | None = None

    iterator: Iterable[int]
    frame_indices = range(start_frame, end_frame + 1)
    if tqdm is not None and total_frames > 0 and not args.verbose:
        iterator = tqdm(frame_indices, desc="Pass 1 poses", unit="frame")
    else:
        iterator = frame_indices

    try:
        for frame_index in iterator:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % args.every != 0:
                continue
            sampled_frames += 1
            if args.max_frames > 0 and sampled_frames > args.max_frames:
                break

            timestamp_sec = frame_index / fps if fps > 0 else 0.0
            color, gray = preprocess_frame(frame_bgr, crop, args.max_dim)
            processed_shape = (color.shape[1], color.shape[0])
            keypoints, descriptors = detect_and_describe(gray, detector)

            if previous_successful_frame is None:
                if descriptors is None or len(keypoints) < args.bootstrap_min_keypoints:
                    pose = FramePose(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        image_path_or_index=frame_index,
                        H_frame_to_world=None,
                        center_world=None,
                        corners_world=None,
                        num_keypoints=len(keypoints),
                        num_matches=0,
                        num_inliers=0,
                        inlier_ratio=0.0,
                        accepted=False,
                        reason=(
                            "waiting for first usable frame "
                            f"(keypoints={len(keypoints)} < {args.bootstrap_min_keypoints})"
                        ),
                    )
                    all_poses.append(pose)
                    if args.verbose:
                        log(f"Skipped bootstrap frame {frame_index}: {pose.reason}", True)
                    continue

                current_frame = ProcessedFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    color=color,
                    gray=gray,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    H_frame_to_world=np.eye(3, dtype=np.float64),
                    tracking_mode="bootstrap",
                )
                if bootstrap_candidate_frame is None:
                    pose = FramePose(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        image_path_or_index=frame_index,
                        H_frame_to_world=None,
                        center_world=None,
                        corners_world=None,
                        num_keypoints=len(keypoints),
                        num_matches=0,
                        num_inliers=0,
                        inlier_ratio=0.0,
                        accepted=False,
                        reason="bootstrap candidate awaiting a matching neighbor",
                    )
                    all_poses.append(pose)
                    bootstrap_candidate_frame = current_frame
                    bootstrap_candidate_pose_index = len(all_poses) - 1
                    if args.verbose:
                        log(
                            f"Stored bootstrap candidate frame {frame_index}: keypoints={len(keypoints)}",
                            True,
                        )
                    continue

                estimate_bootstrap = estimate_pair_transform(
                    bootstrap_candidate_frame,
                    gray,
                    keypoints,
                    descriptors,
                    matcher,
                    args,
                )
                verdict_bootstrap = (
                    (False, estimate_bootstrap.reason)
                    if estimate_bootstrap.H_current_to_reference is None
                    else validate_transform(
                        estimate_bootstrap.H_current_to_reference,
                        estimate_bootstrap,
                        color.shape,
                        args,
                    )
                )
                bootstrap_reliable = (
                    bootstrap_transform_is_reliable(estimate_bootstrap, args)
                    if verdict_bootstrap[0]
                    else (False, verdict_bootstrap[1])
                )
                if verdict_bootstrap[0] and bootstrap_reliable[0]:
                    H_anchor = np.eye(3, dtype=np.float64)
                    anchor_center, anchor_corners = compute_frame_geometry(
                        H_anchor,
                        bootstrap_candidate_frame.color.shape[1],
                        bootstrap_candidate_frame.color.shape[0],
                    )
                    assert bootstrap_candidate_pose_index is not None
                    anchor_pose = all_poses[bootstrap_candidate_pose_index]
                    anchor_pose.H_frame_to_world = H_anchor
                    anchor_pose.center_world = anchor_center
                    anchor_pose.corners_world = anchor_corners
                    anchor_pose.accepted = True
                    anchor_pose.reason = (
                        "bootstrap anchor; matched by frame "
                        f"{frame_index}"
                    )
                    accepted_poses.append(anchor_pose)
                    last_accepted_pose = anchor_pose

                    H_frame_to_world = H_anchor @ estimate_bootstrap.H_current_to_reference
                    center_world, corners_world = compute_frame_geometry(
                        H_frame_to_world,
                        color.shape[1],
                        color.shape[0],
                    )
                    pose_rotation_deg = estimate_rotation_deg_from_H(H_frame_to_world)
                    accepted, reason = should_accept_pose(
                        center_world,
                        pose_rotation_deg,
                        anchor_pose,
                        args,
                    )
                    pose = FramePose(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        image_path_or_index=frame_index,
                        H_frame_to_world=H_frame_to_world,
                        center_world=center_world,
                        corners_world=corners_world,
                        num_keypoints=len(keypoints),
                        num_matches=estimate_bootstrap.num_matches,
                        num_inliers=estimate_bootstrap.num_inliers,
                        inlier_ratio=estimate_bootstrap.inlier_ratio,
                        accepted=accepted,
                        reason=(
                            f"{reason}; bootstrapped from frame "
                            f"{bootstrap_candidate_frame.frame_index}"
                        ),
                    )
                    all_poses.append(pose)
                    current_frame.H_frame_to_world = H_frame_to_world
                    previous_successful_frame = current_frame
                    bootstrap_candidate_frame.tracking_mode = "bootstrap-anchor"
                    bootstrap_candidate_frame.H_frame_to_world = H_anchor
                    append_history_unique(successful_history, bootstrap_candidate_frame)
                    append_history_unique(accepted_frame_history, bootstrap_candidate_frame)
                    if accepted:
                        accepted_poses.append(pose)
                        last_accepted_pose = pose
                        last_accepted_frame = current_frame
                        append_history_unique(accepted_frame_history, current_frame)
                    else:
                        last_accepted_frame = bootstrap_candidate_frame
                    append_history_unique(successful_history, current_frame)
                    consecutive_bridge_frames = 0
                    if args.verbose:
                        log(
                            f"Bootstrapped mapping with frames "
                            f"{bootstrap_candidate_frame.frame_index}->{frame_index}: "
                            f"{estimate_bootstrap.num_inliers}/{estimate_bootstrap.num_matches} inliers",
                            True,
                        )
                    bootstrap_candidate_frame = None
                    bootstrap_candidate_pose_index = None
                    continue

                pose = FramePose(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    image_path_or_index=frame_index,
                    H_frame_to_world=None,
                    center_world=None,
                    corners_world=None,
                    num_keypoints=len(keypoints),
                    num_matches=estimate_bootstrap.num_matches,
                    num_inliers=estimate_bootstrap.num_inliers,
                    inlier_ratio=estimate_bootstrap.inlier_ratio,
                    accepted=False,
                    reason=(
                        "bootstrap pair rejected; "
                        f"{bootstrap_reliable[1]}"
                    ),
                )
                all_poses.append(pose)
                bootstrap_candidate_frame = current_frame
                bootstrap_candidate_pose_index = len(all_poses) - 1
                if args.verbose:
                    log(
                        f"Updated bootstrap candidate to frame {frame_index}: {pose.reason}",
                        True,
                    )
                continue

            predicted_H_frame_to_world = predict_frame_to_world(
                list(successful_history),
                frame_index,
                (color.shape[1], color.shape[0]),
                args,
            )
            estimates: list[tuple[str, ProcessedFrame, TransformEstimate, tuple[bool, str]]] = []
            assert previous_successful_frame is not None

            def evaluate_reference(
                source_name: str,
                reference_frame: ProcessedFrame,
            ) -> tuple[str, ProcessedFrame, TransformEstimate, tuple[bool, str]]:
                H_init_current_to_reference = None
                if predicted_H_frame_to_world is not None:
                    try:
                        H_init_current_to_reference = (
                            np.linalg.inv(reference_frame.H_frame_to_world)
                            @ predicted_H_frame_to_world
                        )
                    except np.linalg.LinAlgError:
                        H_init_current_to_reference = None
                estimate = estimate_pair_transform(
                    reference_frame,
                    gray,
                    keypoints,
                    descriptors,
                    matcher,
                    args,
                    H_init_current_to_reference=H_init_current_to_reference,
                )
                verdict = (
                    (False, estimate.reason)
                    if estimate.H_current_to_reference is None
                    else validate_transform(
                        estimate.H_current_to_reference,
                        estimate,
                        color.shape,
                        args,
                    )
                )
                return (source_name, reference_frame, estimate, verdict)

            estimates.append(evaluate_reference("previous", previous_successful_frame))

            need_reacquire = (
                previous_successful_frame.tracking_mode != "visual"
                or not estimates[0][3][0]
            )
            if need_reacquire:
                for source_name, reference_frame in build_reference_candidates(
                    last_accepted_frame,
                    accepted_frame_history,
                    successful_history,
                    args,
                ):
                    estimates.append(evaluate_reference(source_name, reference_frame))

            valid_estimates = [item for item in estimates if item[3][0]]
            if valid_estimates:
                source_name, reference_frame, best_estimate, _ = max(
                    valid_estimates,
                    key=lambda item: estimate_transform_score(
                        item[2],
                        item[0],
                        frame_index,
                        item[1].frame_index,
                    ),
                )
                assert best_estimate.H_current_to_reference is not None
                H_frame_to_world = (
                    reference_frame.H_frame_to_world @ best_estimate.H_current_to_reference
                )
                center_world, corners_world = compute_frame_geometry(
                    H_frame_to_world,
                    color.shape[1],
                    color.shape[0],
                )
                pose_rotation_deg = estimate_rotation_deg_from_H(H_frame_to_world)
                accepted, reason = should_accept_pose(
                    center_world,
                    pose_rotation_deg,
                    last_accepted_pose if last_accepted_pose is not None else accepted_poses[-1],
                    args,
                )
                pose = FramePose(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    image_path_or_index=frame_index,
                    H_frame_to_world=H_frame_to_world,
                    center_world=center_world,
                    corners_world=corners_world,
                    num_keypoints=len(keypoints),
                    num_matches=best_estimate.num_matches,
                    num_inliers=best_estimate.num_inliers,
                    inlier_ratio=best_estimate.inlier_ratio,
                    accepted=accepted,
                    reason=(
                        f"{reason}; matched to {source_name} frame {reference_frame.frame_index}"
                        + (
                            f" via ECC ({best_estimate.ecc_score:.3f})"
                            if best_estimate.used_ecc and best_estimate.ecc_score is not None
                            else " via optical flow"
                            if best_estimate.used_flow
                            else ""
                        )
                    ),
                )
                all_poses.append(pose)
                current_frame = ProcessedFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    color=color,
                    gray=gray,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    H_frame_to_world=H_frame_to_world,
                    tracking_mode=(
                        "ecc"
                        if best_estimate.used_ecc
                        else "flow"
                        if best_estimate.used_flow
                        else "visual"
                    ),
                )
                previous_successful_frame = current_frame
                append_history_unique(successful_history, current_frame)
                consecutive_bridge_frames = 0
                if accepted:
                    accepted_poses.append(pose)
                    last_accepted_pose = pose
                    last_accepted_frame = current_frame
                    append_history_unique(accepted_frame_history, current_frame)
                    if args.verbose:
                        log(
                            f"Accepted frame {frame_index}: inliers={best_estimate.num_inliers}/"
                            f"{best_estimate.num_matches} ({best_estimate.inlier_ratio:.2f}), {reason}",
                            True,
                        )
                elif args.verbose:
                    log(
                        f"Skipped frame {frame_index}: valid transform but {reason}",
                        True,
                    )
            elif (
                predicted_H_frame_to_world is not None
                and consecutive_bridge_frames < args.max_bridge_frames
            ):
                center_world, corners_world = compute_frame_geometry(
                    predicted_H_frame_to_world,
                    color.shape[1],
                    color.shape[0],
                )
                pose = FramePose(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    image_path_or_index=frame_index,
                    H_frame_to_world=predicted_H_frame_to_world,
                    center_world=center_world,
                    corners_world=corners_world,
                    num_keypoints=len(keypoints),
                    num_matches=0,
                    num_inliers=0,
                    inlier_ratio=0.0,
                    accepted=False,
                    reason=(
                        f"motion bridge {consecutive_bridge_frames + 1}/{args.max_bridge_frames}; "
                        f"predicted from recent successful frames after "
                        f"{'; '.join(f'{name}: {verdict[1]}' for name, _, _, verdict in estimates)}"
                    ),
                )
                all_poses.append(pose)
                current_frame = ProcessedFrame(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    color=color,
                    gray=gray,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    H_frame_to_world=predicted_H_frame_to_world,
                    tracking_mode="predicted",
                )
                previous_successful_frame = current_frame
                append_history_unique(successful_history, current_frame)
                consecutive_bridge_frames += 1
                if args.verbose:
                    log(f"Bridged frame {frame_index}: {pose.reason}", True)
            else:
                reasons = "; ".join(
                    f"{name}: {verdict[1]}"
                    for name, _, estimate, verdict in estimates
                    if estimate is not None
                )
                pose = FramePose(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    image_path_or_index=frame_index,
                    H_frame_to_world=None,
                    center_world=None,
                    corners_world=None,
                    num_keypoints=len(keypoints),
                    num_matches=max((item[2].num_matches for item in estimates), default=0),
                    num_inliers=max((item[2].num_inliers for item in estimates), default=0),
                    inlier_ratio=max((item[2].inlier_ratio for item in estimates), default=0.0),
                    accepted=False,
                    reason=f"transform rejected; {reasons}",
                )
                all_poses.append(pose)
                consecutive_bridge_frames = 0
                if args.verbose:
                    log(f"Rejected frame {frame_index}: {pose.reason}", True)
    finally:
        cap.release()

    if not accepted_poses:
        raise RuntimeError(
            "No frames were accepted. Suggestions: use --every 1, try --detector sift, "
            "reduce --min-step-px, reduce --min-rotation-deg, or add --crop to remove the car body."
        )
    if len(accepted_poses) < 3:
        print(
            "Warning: very few frames were accepted. If the floor is low texture, try "
            "--every 1 --model partial-affine --min-matches 12 --min-inliers 8."
        )
    if processed_shape is None:
        raise RuntimeError("No frames were processed from the video.")
    return all_poses, accepted_poses, processed_shape


def rotation_matrix_about_point(angle_deg: float, center_xy: tuple[float, float]) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = center_xy
    translate_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    rotate = np.array(
        [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    translate_back = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
    return translate_back @ rotate @ translate_to_origin


def apply_presentation_alignment(accepted_poses: list[FramePose]) -> None:
    if len(accepted_poses) < 2:
        return
    first = accepted_poses[0].center_world
    last = accepted_poses[-1].center_world
    if first is None or last is None:
        return
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    if math.hypot(dx, dy) < 1e-6:
        return
    angle_to_right = -math.degrees(math.atan2(dy, dx))
    R = rotation_matrix_about_point(angle_to_right, first)
    for pose in accepted_poses:
        if pose.H_frame_to_world is None or pose.corners_world is None or pose.center_world is None:
            continue
        pose.H_frame_to_world = R @ pose.H_frame_to_world
        pose.center_world = tuple(transform_points(R, np.array([pose.center_world]))[0])
        pose.corners_world = transform_points(R, pose.corners_world)
        pose.reason += "; presentation-aligned"


def compute_world_bounds(
    accepted_poses: Sequence[FramePose],
) -> tuple[float, float, float, float]:
    all_corners = np.vstack([pose.corners_world for pose in accepted_poses if pose.corners_world is not None])
    min_x = float(np.min(all_corners[:, 0]))
    min_y = float(np.min(all_corners[:, 1]))
    max_x = float(np.max(all_corners[:, 0]))
    max_y = float(np.max(all_corners[:, 1]))
    return min_x, min_y, max_x, max_y


def make_world_to_canvas_transform(
    bounds: tuple[float, float, float, float],
    margin_px: int,
) -> tuple[np.ndarray, int, int]:
    min_x, min_y, max_x, max_y = bounds
    width = int(math.ceil(max_x - min_x + 2 * margin_px))
    height = int(math.ceil(max_y - min_y + 2 * margin_px))
    H = np.array(
        [
            [1.0, 0.0, -min_x + margin_px],
            [0.0, 1.0, -min_y + margin_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return H, width, height


def estimate_canvas_memory_bytes(canvas_w: int, canvas_h: int, blend: str) -> int:
    if blend == "last":
        return canvas_w * canvas_h * (3 + 1)
    if blend == "max":
        return canvas_w * canvas_h * (3 + 1 + 1)
    if blend == "best":
        return canvas_w * canvas_h * (3 + 1 + 4)
    if blend == "smart":
        return canvas_w * canvas_h * (3 * 4 + 4 + 3 + 4)
    return canvas_w * canvas_h * (3 * 4 + 4)


def frame_weight_map(size_wh: tuple[int, int], blend: str) -> np.ndarray:
    width, height = size_wh
    if blend == "average":
        return np.ones((height, width), dtype=np.float32)
    if blend in {"feather", "smart"}:
        mask = np.ones((height, width), dtype=np.uint8)
        mask[[0, -1], :] = 0
        mask[:, [0, -1]] = 0
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if dist.max() > 0:
            dist = dist / dist.max()
        dist = np.clip(dist, 1e-3, 1.0)
        return dist.astype(np.float32)
    return np.ones((height, width), dtype=np.float32)


def compute_heat_score(image_bgr: np.ndarray) -> np.ndarray:
    """Compute a scalar score used for overlap selection in `max` blend mode."""

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def normalize_percentile_map(src: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    scale = float(np.percentile(src, percentile))
    if scale <= 1e-6 or not np.isfinite(scale):
        return np.zeros_like(src, dtype=np.float32)
    return np.clip(src / scale, 0.0, 1.0).astype(np.float32)


def compute_detail_score_map(image_bgr: np.ndarray) -> np.ndarray:
    """Estimate a soft detail/saliency map for adaptive blending.

    Flat regions receive low scores and are safe to feather. Regions with strong
    edges or local contrast receive higher scores so the final mosaic keeps a
    single crisp observation instead of averaging them into blur.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    local_contrast = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))

    score = (
        0.55 * normalize_percentile_map(grad_mag)
        + 0.25 * normalize_percentile_map(lap_abs)
        + 0.20 * normalize_percentile_map(local_contrast)
    )
    score = cv2.GaussianBlur(score, (0, 0), 2.0)
    score = cv2.dilate(score, np.ones((9, 9), dtype=np.uint8), iterations=1)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def compose_smart_blend(
    accum_rgb: np.ndarray,
    accum_weight: np.ndarray,
    detail_rgb: np.ndarray,
    detail_score: np.ndarray,
) -> np.ndarray:
    safe_weight = np.where(accum_weight > 1e-6, accum_weight, 1.0)
    base = accum_rgb / safe_weight[:, :, None]
    alpha = np.clip((detail_score - 0.06) / 0.34, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 2.0)
    blended = base * (1.0 - alpha[:, :, None]) + detail_rgb.astype(np.float32) * alpha[:, :, None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def normalize_scalar_series(
    values: Sequence[float],
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, low_percentile))
    hi = float(np.percentile(arr, high_percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return np.full(arr.shape, 0.5, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def estimate_tracking_confidence(pose: FramePose) -> float:
    inlier_term = np.clip((pose.inlier_ratio - 0.15) / 0.55, 0.0, 1.0)
    count_term = np.clip(pose.num_inliers / 30.0, 0.0, 1.0)
    confidence = 0.55 * count_term + 0.45 * inlier_term

    reason = pose.reason.lower()
    if "motion bridge" in reason:
        confidence *= 0.70
    elif "ecc" in reason:
        confidence *= 0.90
    elif "optical flow" in reason:
        confidence *= 0.95
    return float(np.clip(confidence, 0.0, 1.0))


def analyze_accepted_frame_content(
    accepted_poses: Sequence[FramePose],
    frame_paths: dict[int, Path],
) -> dict[int, FrameContentAnalysis]:
    """Analyze all accepted frames before rendering to bias overlap selection.

    The current pipeline already estimates geometry first and renders second.
    This pass makes the content decision global as well by ranking accepted
    frames using sharpness, detail, and tracking confidence before stitching.
    """

    if not accepted_poses:
        return {}

    sharpness_values: list[float] = []
    detail_values: list[float] = []
    tracking_values: list[float] = []
    raw_metrics: list[tuple[FramePose, float, float, float]] = []

    for pose in accepted_poses:
        image = load_cached_frame(frame_paths[pose.frame_index])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        detail_mean = float(np.mean(compute_detail_score_map(image)))
        tracking_confidence = estimate_tracking_confidence(pose)
        sharpness_values.append(sharpness)
        detail_values.append(detail_mean)
        tracking_values.append(tracking_confidence)
        raw_metrics.append((pose, sharpness, detail_mean, tracking_confidence))

    sharpness_norm = normalize_scalar_series(sharpness_values)
    detail_norm = normalize_scalar_series(detail_values)
    tracking_norm = normalize_scalar_series(tracking_values, 0.0, 100.0)

    analysis: dict[int, FrameContentAnalysis] = {}
    for idx, (pose, sharpness, detail_mean, tracking_confidence) in enumerate(raw_metrics):
        combined = (
            0.45 * float(sharpness_norm[idx])
            + 0.35 * float(detail_norm[idx])
            + 0.20 * float(tracking_norm[idx])
        )
        global_weight = 0.85 + 0.30 * combined
        analysis[pose.frame_index] = FrameContentAnalysis(
            frame_index=pose.frame_index,
            sharpness=sharpness,
            detail_mean=detail_mean,
            tracking_confidence=tracking_confidence,
            global_weight=float(global_weight),
        )
    return analysis


def smart_weight_gain(
    frame_analysis: dict[int, FrameContentAnalysis] | None,
    frame_index: int,
) -> float:
    if not frame_analysis:
        return 1.0
    analysis = frame_analysis.get(frame_index)
    if analysis is None:
        return 1.0
    return analysis.global_weight


def compute_selection_score_map(
    image_bgr: np.ndarray,
    quality_gain: float,
) -> np.ndarray:
    detail = compute_detail_score_map(image_bgr)
    feather = frame_weight_map((image_bgr.shape[1], image_bgr.shape[0]), "feather")
    score = feather * (0.18 + 0.82 * detail) * quality_gain
    return np.clip(score, 0.0, 2.0).astype(np.float32)


def update_best_detail_layer(
    detail_rgb: np.ndarray,
    detail_score: np.ndarray,
    warped_rgb: np.ndarray,
    detail_candidate: np.ndarray,
    mask_bool: np.ndarray,
) -> None:
    better_mask = mask_bool & (detail_candidate >= detail_score)
    if not np.any(better_mask):
        return
    detail_rgb[better_mask] = warped_rgb[better_mask]
    detail_score[better_mask] = detail_candidate[better_mask]


def pose_to_serializable(pose: FramePose) -> dict:
    data = asdict(pose)
    if pose.H_frame_to_world is not None:
        data["H_frame_to_world"] = pose.H_frame_to_world.tolist()
    if pose.corners_world is not None:
        data["corners_world"] = pose.corners_world.tolist()
    return data


def write_pose_log_csv(path: Path, poses: Sequence[FramePose]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp_sec",
                "accepted",
                "reason",
                "num_keypoints",
                "num_matches",
                "num_inliers",
                "inlier_ratio",
                "center_x",
                "center_y",
                "h00",
                "h01",
                "h02",
                "h10",
                "h11",
                "h12",
                "h20",
                "h21",
                "h22",
            ]
        )
        for pose in poses:
            H = pose.H_frame_to_world if pose.H_frame_to_world is not None else np.full((3, 3), np.nan)
            center_x, center_y = pose.center_world if pose.center_world is not None else (math.nan, math.nan)
            writer.writerow(
                [
                    pose.frame_index,
                    f"{pose.timestamp_sec:.6f}",
                    int(pose.accepted),
                    pose.reason,
                    pose.num_keypoints,
                    pose.num_matches,
                    pose.num_inliers,
                    f"{pose.inlier_ratio:.6f}",
                    center_x,
                    center_y,
                    *H.reshape(-1).tolist(),
                ]
            )


def write_pose_log_json(path: Path, poses: Sequence[FramePose]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([pose_to_serializable(pose) for pose in poses], handle, indent=2)


def cache_accepted_frames(
    video_path: Path,
    accepted_poses: Sequence[FramePose],
    crop: tuple[int, int, int, int] | None,
    args: argparse.Namespace,
) -> tuple[tempfile.TemporaryDirectory[str], dict[int, Path], dict[int, Path]]:
    """Decode accepted frames once and cache processed and optional raw imagery."""

    temp_dir = tempfile.TemporaryDirectory(prefix="floor_video_map_")
    cache_root = Path(temp_dir.name)
    accepted_indices = {pose.frame_index for pose in accepted_poses}
    frame_paths: dict[int, Path] = {}
    raw_frame_paths: dict[int, Path] = {}

    cap = open_capture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame, end_frame = resolve_frame_range(frame_count, args)
    if accepted_indices:
        start_frame = max(start_frame, min(accepted_indices))
        end_frame = min(end_frame, max(accepted_indices))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    iterator: Iterable[int]
    frame_indices = range(start_frame, end_frame + 1)
    if tqdm is not None and frame_count > 0 and not args.verbose:
        iterator = tqdm(frame_indices, desc="Cache frames", unit="frame")
    else:
        iterator = frame_indices
    try:
        for frame_index in iterator:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index not in accepted_indices:
                continue
            color, _ = preprocess_frame(frame_bgr, crop, args.max_dim)
            out_path = cache_root / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(out_path), color):
                raise RuntimeError(f"Failed to write temporary cached frame: {out_path}")
            frame_paths[frame_index] = out_path
            if args.comparison_video or args.comparison_edge_video:
                raw_out_path = cache_root / f"raw_{frame_index:06d}.jpg"
                if not cv2.imwrite(
                    str(raw_out_path),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(f"Failed to write temporary cached frame: {raw_out_path}")
                raw_frame_paths[frame_index] = raw_out_path
    finally:
        cap.release()

    missing = sorted(accepted_indices - set(frame_paths))
    if missing:
        raise RuntimeError(f"Missing cached accepted frames: {missing[:10]}")
    return temp_dir, frame_paths, raw_frame_paths


def load_cached_frame(frame_path: Path) -> np.ndarray:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load cached frame: {frame_path}")
    return image


def crop_to_valid_region(
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    coords = np.column_stack(np.where(valid_mask > 0))
    if coords.size == 0:
        return image_bgr, valid_mask, (0, 0, image_bgr.shape[1], image_bgr.shape[0])
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped_image = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    cropped_mask = valid_mask[y0 : y1 + 1, x0 : x1 + 1]
    return cropped_image, cropped_mask, (int(x0), int(y0), int(x1 + 1), int(y1 + 1))


def ensure_parent_dir(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def resolve_preview_blend(args: argparse.Namespace) -> str:
    return args.blend if args.preview_blend == "inherit" else args.preview_blend


def put_text_lines(
    image_bgr: np.ndarray,
    lines: Sequence[str],
    origin_xy: tuple[int, int] = (12, 24),
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = origin_xy
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22


def build_overlay_lines(
    pose: FramePose,
    canvas_shape: tuple[int, int],
) -> list[str]:
    lines = [f"accepted frame {pose.frame_index}  t={pose.timestamp_sec:.2f}s"]
    lines.append(f"inliers {pose.num_inliers}/{pose.num_matches}  ratio {pose.inlier_ratio:.2f}")
    lines.append(f"canvas {canvas_shape[1]}x{canvas_shape[0]}")
    lines.append(pose.reason)
    return lines


def transform_canvas_points(
    points_world: np.ndarray,
    T_world_to_canvas: np.ndarray,
) -> np.ndarray:
    return transform_points(T_world_to_canvas, points_world)


def draw_debug_overlay(
    base_bgr: np.ndarray,
    accepted_poses: Sequence[FramePose],
    current_pose: FramePose,
    T_world_to_canvas: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    vis = base_bgr.copy()
    if args.draw_trajectory:
        centers = np.array(
            [pose.center_world for pose in accepted_poses if pose.center_world is not None],
            dtype=np.float64,
        )
        if len(centers) >= 2:
            centers_canvas = transform_canvas_points(centers, T_world_to_canvas).astype(np.int32)
            cv2.polylines(vis, [centers_canvas], False, (0, 255, 255), 2, cv2.LINE_AA)
    if args.draw_current_frame_outline and current_pose.corners_world is not None:
        polygon = transform_canvas_points(current_pose.corners_world, T_world_to_canvas).astype(np.int32)
        cv2.polylines(vis, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
    if current_pose.center_world is not None:
        center_canvas = transform_canvas_points(
            np.array([current_pose.center_world], dtype=np.float64),
            T_world_to_canvas,
        )[0]
        cv2.drawMarker(
            vis,
            (int(round(center_canvas[0])), int(round(center_canvas[1]))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
    lines: list[str] = []
    if args.draw_frame_index or args.draw_quality:
        lines = build_overlay_lines(current_pose, vis.shape[:2])
    if lines:
        put_text_lines(vis, lines)
    return vis


def scale_and_letterbox(
    image_bgr: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    src_h, src_w = image_bgr.shape[:2]
    scale = min(width / max(1, src_w), height / max(1, src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_w) // 2
    y0 = (height - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def draw_panel_label(image_bgr: np.ndarray, label: str) -> None:
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 34), (0, 0, 0), thickness=-1)
    put_text_lines(image_bgr, [label], origin_xy=(10, 24), color=(255, 255, 255))


def compute_frame_edges(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    median = float(np.median(gray))
    low = int(max(10, 0.66 * median))
    high = int(min(255, max(low + 20, 1.33 * median + 20)))
    edges = cv2.Canny(gray, low, high, L2gradient=True)
    return cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)


def colorize_edge_canvas(edge_gray: np.ndarray) -> np.ndarray:
    glow = cv2.GaussianBlur(edge_gray, (0, 0), 1.2)
    edge_vis = np.zeros((edge_gray.shape[0], edge_gray.shape[1], 3), dtype=np.uint8)
    edge_vis[:, :, 0] = np.clip(glow * 0.80, 0, 255).astype(np.uint8)
    edge_vis[:, :, 1] = np.clip(glow * 0.95 + edge_gray * 0.35, 0, 255).astype(np.uint8)
    edge_vis[:, :, 2] = np.clip(edge_gray * 0.20, 0, 255).astype(np.uint8)
    return edge_vis


def make_progress_video_frame(
    current_canvas_bgr: np.ndarray,
    valid_mask: np.ndarray,
    accepted_prefix: Sequence[FramePose],
    current_pose: FramePose,
    T_world_to_canvas: np.ndarray,
    args: argparse.Namespace,
    output_width: int | None = None,
    output_height: int | None = None,
) -> np.ndarray:
    del valid_mask  # reserved for future view-dependent cropping
    output_width = output_width or args.video_width
    output_height = output_height or args.video_height

    if args.draw_trajectory or args.draw_current_frame_outline or args.draw_frame_index or args.draw_quality:
        overlay_canvas = draw_debug_overlay(
            current_canvas_bgr,
            accepted_prefix,
            current_pose,
            T_world_to_canvas,
            args,
        )
    else:
        overlay_canvas = current_canvas_bgr

    if args.video_view == "canvas":
        if overlay_canvas.shape[1] == output_width and overlay_canvas.shape[0] == output_height:
            return overlay_canvas
        return scale_and_letterbox(overlay_canvas, output_width, output_height)

    if args.video_view == "overview":
        return scale_and_letterbox(overlay_canvas, output_width, output_height)

    if args.video_view == "follow":
        if current_pose.center_world is None:
            return scale_and_letterbox(overlay_canvas, output_width, output_height)
        center_canvas = transform_canvas_points(
            np.array([current_pose.center_world], dtype=np.float64),
            T_world_to_canvas,
        )[0]
        half_w = max(1, int(round(output_width * 0.5 / max(args.follow_scale, 1e-6))))
        half_h = max(1, int(round(output_height * 0.5 / max(args.follow_scale, 1e-6))))
        cx = int(round(center_canvas[0]))
        cy = int(round(center_canvas[1]))
        x0 = cx - half_w
        y0 = cy - half_h
        x1 = cx + half_w
        y1 = cy + half_h
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - overlay_canvas.shape[0])
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - overlay_canvas.shape[1])
        padded = cv2.copyMakeBorder(
            overlay_canvas,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
        crop = padded[y0:y1, x0:x1]
        return cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_AREA)

    raise ValueError(f"Unsupported video view: {args.video_view}")


def render_trajectory_panel(
    accepted_prefix: Sequence[FramePose],
    current_pose: FramePose,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    margin_px: int,
) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    T_world_to_bounds, world_w, world_h = make_world_to_canvas_transform(bounds, margin_px)
    draw_scale = min(
        max(1.0, width - 24) / max(1, world_w),
        max(1.0, height - 24) / max(1, world_h),
    )
    draw_w = max(1, int(round(world_w * draw_scale)))
    draw_h = max(1, int(round(world_h * draw_scale)))
    offset_x = max(0, (width - draw_w) // 2)
    offset_y = max(0, (height - draw_h) // 2)
    S = np.array(
        [[draw_scale, 0.0, offset_x], [0.0, draw_scale, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_world_to_panel = S @ T_world_to_bounds

    centers = np.array(
        [pose.center_world for pose in accepted_prefix if pose.center_world is not None],
        dtype=np.float64,
    )
    if len(centers) >= 2:
        centers_panel = transform_canvas_points(centers, T_world_to_panel).astype(np.int32)
        cv2.polylines(panel, [centers_panel], False, (32, 180, 255), 2, cv2.LINE_AA)

    step = max(1, len(accepted_prefix) // 24)
    for pose in accepted_prefix[::step]:
        if pose.corners_world is None:
            continue
        polygon = transform_canvas_points(pose.corners_world, T_world_to_panel).astype(np.int32)
        cv2.polylines(panel, [polygon], True, (0, 180, 0), 1, cv2.LINE_AA)

    if current_pose.corners_world is not None:
        polygon = transform_canvas_points(current_pose.corners_world, T_world_to_panel).astype(np.int32)
        cv2.polylines(panel, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)

    if current_pose.center_world is not None:
        center_panel = transform_canvas_points(
            np.array([current_pose.center_world], dtype=np.float64),
            T_world_to_panel,
        )[0]
        cv2.drawMarker(
            panel,
            (int(round(center_panel[0])), int(round(center_panel[1]))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    lines = [
        f"accepted {len(accepted_prefix)}",
        f"frame {current_pose.frame_index}",
        f"t={current_pose.timestamp_sec:.2f}s",
    ]
    put_text_lines(panel, lines, origin_xy=(10, height - 48))
    return panel


def build_comparison_frame(
    original_bgr: np.ndarray,
    current_canvas_bgr: np.ndarray,
    valid_mask: np.ndarray,
    accepted_prefix: Sequence[FramePose],
    current_pose: FramePose,
    T_world_to_canvas: np.ndarray,
    bounds: tuple[float, float, float, float],
    args: argparse.Namespace,
    middle_canvas_bgr: np.ndarray | None = None,
    middle_label: str = "Mosaic",
) -> np.ndarray:
    left_w = args.video_width // 3
    middle_w = args.video_width // 3
    right_w = args.video_width - left_w - middle_w

    left_panel = scale_and_letterbox(original_bgr, left_w, args.video_height)
    middle_source = current_canvas_bgr if middle_canvas_bgr is None else middle_canvas_bgr
    middle_panel = make_progress_video_frame(
        middle_source,
        valid_mask,
        accepted_prefix,
        current_pose,
        T_world_to_canvas,
        args,
        output_width=middle_w,
        output_height=args.video_height,
    )
    right_panel = render_trajectory_panel(
        accepted_prefix,
        current_pose,
        bounds,
        right_w,
        args.video_height,
        args.margin_px,
    )

    draw_panel_label(left_panel, "Original")
    draw_panel_label(middle_panel, middle_label)
    draw_panel_label(right_panel, "Trajectory")

    comparison = np.hstack([left_panel, middle_panel, right_panel])
    separator_x = left_w
    cv2.line(comparison, (separator_x, 0), (separator_x, comparison.shape[0]), (48, 48, 48), 2)
    separator_x = left_w + middle_w
    cv2.line(comparison, (separator_x, 0), (separator_x, comparison.shape[0]), (48, 48, 48), 2)
    return comparison


def render_progress_video(
    current_canvas_bgr: np.ndarray,
    valid_mask: np.ndarray,
    accepted_prefix: Sequence[FramePose],
    current_pose: FramePose,
    T_world_to_canvas: np.ndarray,
    writer: cv2.VideoWriter,
    args: argparse.Namespace,
) -> None:
    frame_vis = make_progress_video_frame(
        current_canvas_bgr,
        valid_mask,
        accepted_prefix,
        current_pose,
        T_world_to_canvas,
        args,
    )
    writer.write(frame_vis)


def render_mosaic_single_canvas(
    accepted_poses: Sequence[FramePose],
    frame_paths: dict[int, Path],
    raw_frame_paths: dict[int, Path],
    frame_analysis: dict[int, FrameContentAnalysis] | None,
    bounds: tuple[float, float, float, float],
    T_world_to_canvas: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    args: argparse.Namespace,
) -> RenderResult:
    """Render all accepted frames into a single canvas."""

    if args.video_view == "canvas" and (
        args.output_video or args.comparison_video or args.comparison_edge_video
    ):
        if canvas_w > args.video_width or canvas_h > args.video_height:
            print(
                "Warning: --video-view canvas ignores --video-width/--video-height and uses the full canvas size."
            )

    if args.blend == "last":
        mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        accum_rgb = None
        accum_weight = None
        max_score = None
        best_score = None
        detail_rgb = None
        detail_score = None
    elif args.blend == "max":
        mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        max_score = np.full((canvas_h, canvas_w), -1, dtype=np.int16)
        accum_rgb = None
        accum_weight = None
        best_score = None
        detail_rgb = None
        detail_score = None
    elif args.blend == "best":
        mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        best_score = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
        accum_rgb = None
        accum_weight = None
        max_score = None
        detail_rgb = None
        detail_score = None
    elif args.blend == "smart":
        accum_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        detail_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        detail_score = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        mosaic = None
        valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        max_score = None
        best_score = None
    else:
        accum_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        mosaic = None
        valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        max_score = None
        best_score = None
        detail_rgb = None
        detail_score = None

    writer: cv2.VideoWriter | None = None
    comparison_writer: cv2.VideoWriter | None = None
    comparison_edge_writer: cv2.VideoWriter | None = None
    edge_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    if args.output_video:
        ensure_parent_dir(Path(args.output_video).resolve())
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        video_fps = args.video_fps if args.video_fps > 0 else 20.0
        if args.video_view == "canvas":
            video_size = (canvas_w, canvas_h)
        else:
            video_size = (args.video_width, args.video_height)
        writer = cv2.VideoWriter(str(args.output_video), fourcc, video_fps, video_size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output video for writing: {args.output_video}")
    if args.comparison_video:
        ensure_parent_dir(Path(args.comparison_video).resolve())
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        video_fps = args.video_fps if args.video_fps > 0 else 20.0
        comparison_writer = cv2.VideoWriter(
            str(args.comparison_video),
            fourcc,
            video_fps,
            (args.video_width, args.video_height),
        )
        if not comparison_writer.isOpened():
            raise RuntimeError(
                f"Failed to open comparison video for writing: {args.comparison_video}"
            )
    if args.comparison_edge_video:
        ensure_parent_dir(Path(args.comparison_edge_video).resolve())
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        video_fps = args.video_fps if args.video_fps > 0 else 20.0
        comparison_edge_writer = cv2.VideoWriter(
            str(args.comparison_edge_video),
            fourcc,
            video_fps,
            (args.video_width, args.video_height),
        )
        if not comparison_edge_writer.isOpened():
            raise RuntimeError(
                f"Failed to open edge comparison video for writing: {args.comparison_edge_video}"
            )

    accepted_prefix: list[FramePose] = []
    last_raw_image: np.ndarray | None = None
    try:
        iterator: Iterable[FramePose]
        if tqdm is not None and not args.verbose:
            iterator = tqdm(accepted_poses, desc="Render mosaic", unit="frame")
        else:
            iterator = accepted_poses
        for pose in iterator:
            frame_path = frame_paths[pose.frame_index]
            image = load_cached_frame(frame_path)
            raw_image = None
            if comparison_writer is not None or comparison_edge_writer is not None:
                raw_frame_path = raw_frame_paths.get(pose.frame_index)
                raw_image = load_cached_frame(raw_frame_path) if raw_frame_path is not None else image
                last_raw_image = raw_image
            H_frame_to_canvas = T_world_to_canvas @ pose.H_frame_to_world
            warp_size = (canvas_w, canvas_h)
            warped = cv2.warpPerspective(
                image,
                H_frame_to_canvas,
                warp_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask_src = np.full(image.shape[:2], 255, dtype=np.uint8)
            warped_mask = cv2.warpPerspective(
                mask_src,
                H_frame_to_canvas,
                warp_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask_bool = warped_mask > 0
            if args.blend == "last":
                mosaic[mask_bool] = warped[mask_bool]
                valid_mask[mask_bool] = 255
                current_canvas = mosaic
            elif args.blend == "max":
                warped_score = compute_heat_score(warped).astype(np.int16)
                update_mask = mask_bool & (warped_score >= max_score)
                mosaic[update_mask] = warped[update_mask]
                max_score[update_mask] = warped_score[update_mask]
                valid_mask[mask_bool] = 255
                current_canvas = mosaic
            elif args.blend == "best":
                quality_gain = smart_weight_gain(frame_analysis, pose.frame_index)
                score_src = compute_selection_score_map(image, quality_gain)
                warped_score = cv2.warpPerspective(
                    score_src,
                    H_frame_to_canvas,
                    warp_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                warped_score *= mask_bool.astype(np.float32)
                update_mask = mask_bool & (warped_score >= best_score)
                mosaic[update_mask] = warped[update_mask]
                best_score[update_mask] = warped_score[update_mask]
                valid_mask[mask_bool] = 255
                current_canvas = mosaic
            elif args.blend == "smart":
                weight_src = frame_weight_map((image.shape[1], image.shape[0]), "feather")
                detail_src = compute_detail_score_map(image)
                quality_gain = smart_weight_gain(frame_analysis, pose.frame_index)
                warped_weight = cv2.warpPerspective(
                    weight_src,
                    H_frame_to_canvas,
                    warp_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                warped_detail = cv2.warpPerspective(
                    detail_src,
                    H_frame_to_canvas,
                    warp_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                warped_weight *= mask_bool.astype(np.float32)
                warped_detail *= mask_bool.astype(np.float32)
                warped_weight *= quality_gain
                accum_rgb += warped.astype(np.float32) * warped_weight[:, :, None]
                accum_weight += warped_weight
                detail_candidate = np.clip(
                    warped_detail * np.clip(warped_weight, 0.25, 1.0) * quality_gain,
                    0.0,
                    1.0,
                )
                update_best_detail_layer(
                    detail_rgb,
                    detail_score,
                    warped,
                    detail_candidate,
                    mask_bool,
                )
                valid_mask[accum_weight > 0] = 255
                current_canvas = compose_smart_blend(
                    accum_rgb,
                    accum_weight,
                    detail_rgb,
                    detail_score,
                )
            else:
                weight_src = frame_weight_map((image.shape[1], image.shape[0]), args.blend)
                warped_weight = cv2.warpPerspective(
                    weight_src,
                    H_frame_to_canvas,
                    warp_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                warped_weight *= mask_bool.astype(np.float32)
                accum_rgb += warped.astype(np.float32) * warped_weight[:, :, None]
                accum_weight += warped_weight
                valid_mask[accum_weight > 0] = 255
                current_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                safe_weight = np.where(accum_weight > 1e-6, accum_weight, 1.0)
                current_canvas = np.clip(
                    accum_rgb / safe_weight[:, :, None],
                    0,
                    255,
                ).astype(np.uint8)
            accepted_prefix.append(pose)
            if writer is not None:
                render_progress_video(
                    current_canvas,
                    valid_mask,
                    accepted_prefix,
                    pose,
                    T_world_to_canvas,
                    writer,
                    args,
                )
            if comparison_writer is not None and raw_image is not None:
                comparison_frame = build_comparison_frame(
                    raw_image,
                    current_canvas,
                    valid_mask,
                    accepted_prefix,
                    pose,
                    T_world_to_canvas,
                    bounds,
                    args,
                )
                comparison_writer.write(comparison_frame)
            if comparison_edge_writer is not None and raw_image is not None:
                edge_src = compute_frame_edges(image)
                warped_edges = cv2.warpPerspective(
                    edge_src,
                    H_frame_to_canvas,
                    warp_size,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                edge_canvas = np.maximum(edge_canvas, warped_edges)
                edge_frame = build_comparison_frame(
                    raw_image,
                    current_canvas,
                    valid_mask,
                    accepted_prefix,
                    pose,
                    T_world_to_canvas,
                    bounds,
                    args,
                    middle_canvas_bgr=colorize_edge_canvas(edge_canvas),
                    middle_label="Edges",
                )
                comparison_edge_writer.write(edge_frame)
    finally:
        if writer is not None:
            writer.release()
        if comparison_writer is not None:
            comparison_writer.release()

    if args.blend in {"last", "max", "best"}:
        mosaic_bgr = mosaic
    elif args.blend == "smart":
        mosaic_bgr = compose_smart_blend(
            accum_rgb,
            accum_weight,
            detail_rgb,
            detail_score,
        )
    else:
        safe_weight = np.where(accum_weight > 1e-6, accum_weight, 1.0)
        mosaic_bgr = np.clip(accum_rgb / safe_weight[:, :, None], 0, 255).astype(np.uint8)

    final_canvas_bgr = mosaic_bgr.copy()
    final_valid_mask = valid_mask.copy()

    if comparison_edge_writer is not None and accepted_poses and last_raw_image is not None:
        hold_frames = max(1, int(round(args.video_fps * max(args.edge_final_hold_sec, 0.1))))
        final_pose = accepted_poses[-1]
        final_comparison_frame = build_comparison_frame(
            last_raw_image,
            final_canvas_bgr,
            final_valid_mask,
            accepted_poses,
            final_pose,
            T_world_to_canvas,
            bounds,
            args,
            middle_canvas_bgr=final_canvas_bgr,
            middle_label="Final Mosaic",
        )
        for _ in range(hold_frames):
            comparison_edge_writer.write(final_comparison_frame)
        comparison_edge_writer.release()
        comparison_edge_writer = None

    if args.debug_overlay_on_mosaic and accepted_poses:
        mosaic_bgr = draw_debug_overlay(
            mosaic_bgr,
            accepted_poses,
            accepted_poses[-1],
            T_world_to_canvas,
            args,
        )

    crop_box: tuple[int, int, int, int] | None = None
    if not args.no_crop_output:
        mosaic_bgr, valid_mask, crop_box = crop_to_valid_region(mosaic_bgr, valid_mask)

    return RenderResult(mosaic_bgr=mosaic_bgr, valid_mask=valid_mask, crop_box=crop_box)


def preview_transform(
    T_world_to_canvas: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    preview_max_dim: int = 1600,
) -> tuple[np.ndarray, int, int, float]:
    scale = min(1.0, preview_max_dim / max(canvas_w, canvas_h))
    preview_w = max(1, int(math.ceil(canvas_w * scale)))
    preview_h = max(1, int(math.ceil(canvas_h * scale)))
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return S @ T_world_to_canvas, preview_w, preview_h, scale


def render_preview_canvas(
    accepted_poses: Sequence[FramePose],
    frame_paths: dict[int, Path],
    T_world_to_canvas: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    blend: str,
    preview_path: Path,
    frame_analysis: dict[int, FrameContentAnalysis] | None = None,
) -> None:
    preview_transform_matrix, preview_w, preview_h, _ = preview_transform(
        T_world_to_canvas,
        canvas_w,
        canvas_h,
    )
    if blend == "last":
        mosaic = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        valid_mask = np.zeros((preview_h, preview_w), dtype=np.uint8)
        for pose in accepted_poses:
            image = load_cached_frame(frame_paths[pose.frame_index])
            H = preview_transform_matrix @ pose.H_frame_to_world
            warped = cv2.warpPerspective(image, H, (preview_w, preview_h))
            mask = cv2.warpPerspective(
                np.full(image.shape[:2], 255, dtype=np.uint8),
                H,
                (preview_w, preview_h),
                flags=cv2.INTER_NEAREST,
            )
            mask_bool = mask > 0
            mosaic[mask_bool] = warped[mask_bool]
            valid_mask[mask_bool] = 255
        preview_bgr, _, _ = crop_to_valid_region(mosaic, valid_mask)
    elif blend == "best":
        mosaic = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        valid_mask = np.zeros((preview_h, preview_w), dtype=np.uint8)
        best_score = np.full((preview_h, preview_w), -1.0, dtype=np.float32)
        for pose in accepted_poses:
            image = load_cached_frame(frame_paths[pose.frame_index])
            H = preview_transform_matrix @ pose.H_frame_to_world
            quality_gain = smart_weight_gain(frame_analysis, pose.frame_index)
            warped = cv2.warpPerspective(image, H, (preview_w, preview_h))
            mask = cv2.warpPerspective(
                np.full(image.shape[:2], 255, dtype=np.uint8),
                H,
                (preview_w, preview_h),
                flags=cv2.INTER_NEAREST,
            )
            mask_bool = mask > 0
            score = cv2.warpPerspective(
                compute_selection_score_map(image, quality_gain),
                H,
                (preview_w, preview_h),
                flags=cv2.INTER_LINEAR,
            )
            score *= mask_bool.astype(np.float32)
            update_mask = mask_bool & (score >= best_score)
            mosaic[update_mask] = warped[update_mask]
            best_score[update_mask] = score[update_mask]
            valid_mask[mask_bool] = 255
        preview_bgr, _, _ = crop_to_valid_region(mosaic, valid_mask)
    elif blend == "max":
        mosaic = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        valid_mask = np.zeros((preview_h, preview_w), dtype=np.uint8)
        max_score = np.full((preview_h, preview_w), -1, dtype=np.int16)
        for pose in accepted_poses:
            image = load_cached_frame(frame_paths[pose.frame_index])
            H = preview_transform_matrix @ pose.H_frame_to_world
            warped = cv2.warpPerspective(image, H, (preview_w, preview_h))
            mask = cv2.warpPerspective(
                np.full(image.shape[:2], 255, dtype=np.uint8),
                H,
                (preview_w, preview_h),
                flags=cv2.INTER_NEAREST,
            )
            mask_bool = mask > 0
            warped_score = compute_heat_score(warped).astype(np.int16)
            update_mask = mask_bool & (warped_score >= max_score)
            mosaic[update_mask] = warped[update_mask]
            max_score[update_mask] = warped_score[update_mask]
            valid_mask[mask_bool] = 255
        preview_bgr, _, _ = crop_to_valid_region(mosaic, valid_mask)
    elif blend == "smart":
        accum_rgb = np.zeros((preview_h, preview_w, 3), dtype=np.float32)
        accum_weight = np.zeros((preview_h, preview_w), dtype=np.float32)
        detail_rgb = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        detail_score = np.zeros((preview_h, preview_w), dtype=np.float32)
        valid_mask = np.zeros((preview_h, preview_w), dtype=np.uint8)
        for pose in accepted_poses:
            image = load_cached_frame(frame_paths[pose.frame_index])
            H = preview_transform_matrix @ pose.H_frame_to_world
            quality_gain = smart_weight_gain(frame_analysis, pose.frame_index)
            warped = cv2.warpPerspective(image, H, (preview_w, preview_h))
            mask = cv2.warpPerspective(
                np.full(image.shape[:2], 255, dtype=np.uint8),
                H,
                (preview_w, preview_h),
                flags=cv2.INTER_NEAREST,
            )
            mask_bool = mask > 0
            weight = cv2.warpPerspective(
                frame_weight_map((image.shape[1], image.shape[0]), "feather"),
                H,
                (preview_w, preview_h),
            )
            detail = cv2.warpPerspective(
                compute_detail_score_map(image),
                H,
                (preview_w, preview_h),
            )
            weight *= mask_bool.astype(np.float32)
            detail *= mask_bool.astype(np.float32)
            weight *= quality_gain
            accum_rgb += warped.astype(np.float32) * weight[:, :, None]
            accum_weight += weight
            detail_candidate = np.clip(
                detail * np.clip(weight, 0.25, 1.0) * quality_gain,
                0.0,
                1.0,
            )
            update_best_detail_layer(
                detail_rgb,
                detail_score,
                warped,
                detail_candidate,
                mask_bool,
            )
            valid_mask[accum_weight > 0] = 255
        preview_bgr = compose_smart_blend(
            accum_rgb,
            accum_weight,
            detail_rgb,
            detail_score,
        )
        preview_bgr, _, _ = crop_to_valid_region(preview_bgr, valid_mask)
    else:
        accum_rgb = np.zeros((preview_h, preview_w, 3), dtype=np.float32)
        accum_weight = np.zeros((preview_h, preview_w), dtype=np.float32)
        for pose in accepted_poses:
            image = load_cached_frame(frame_paths[pose.frame_index])
            H = preview_transform_matrix @ pose.H_frame_to_world
            warped = cv2.warpPerspective(image, H, (preview_w, preview_h))
            weight = cv2.warpPerspective(
                frame_weight_map((image.shape[1], image.shape[0]), blend),
                H,
                (preview_w, preview_h),
            )
            accum_rgb += warped.astype(np.float32) * weight[:, :, None]
            accum_weight += weight
        safe_weight = np.where(accum_weight > 1e-6, accum_weight, 1.0)
        preview_bgr = np.clip(accum_rgb / safe_weight[:, :, None], 0, 255).astype(np.uint8)
    ensure_parent_dir(preview_path)
    if not cv2.imwrite(str(preview_path), preview_bgr):
        raise RuntimeError(f"Failed to write preview image: {preview_path}")


def tile_intersects_pose(
    pose: FramePose,
    T_world_to_canvas: np.ndarray,
    tile_bounds: tuple[int, int, int, int],
) -> bool:
    if pose.corners_world is None:
        return False
    x0, y0, x1, y1 = tile_bounds
    corners_canvas = transform_canvas_points(pose.corners_world, T_world_to_canvas)
    min_x = float(np.min(corners_canvas[:, 0]))
    min_y = float(np.min(corners_canvas[:, 1]))
    max_x = float(np.max(corners_canvas[:, 0]))
    max_y = float(np.max(corners_canvas[:, 1]))
    return not (max_x < x0 or max_y < y0 or min_x > x1 or min_y > y1)


def render_mosaic_tiles(
    accepted_poses: Sequence[FramePose],
    frame_paths: dict[int, Path],
    frame_analysis: dict[int, FrameContentAnalysis] | None,
    video_path: Path,
    bounds: tuple[float, float, float, float],
    T_world_to_canvas: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    args: argparse.Namespace,
) -> None:
    tile_dir = Path(args.tile_output_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    manifest_tiles: list[dict] = []
    preview_blend = resolve_preview_blend(args)

    for tile_y0 in range(0, canvas_h, args.tile_size):
        for tile_x0 in range(0, canvas_w, args.tile_size):
            tile_w = min(args.tile_size, canvas_w - tile_x0)
            tile_h = min(args.tile_size, canvas_h - tile_y0)
            tile_bounds = (tile_x0, tile_y0, tile_x0 + tile_w, tile_y0 + tile_h)
            relevant_poses = [
                pose
                for pose in accepted_poses
                if tile_intersects_pose(pose, T_world_to_canvas, tile_bounds)
            ]
            if not relevant_poses:
                continue

            if args.blend == "last":
                tile_image = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                tile_valid = np.zeros((tile_h, tile_w), dtype=np.uint8)
                tile_accum_rgb = None
                tile_accum_weight = None
                tile_max_score = None
                tile_best_score = None
                tile_detail_rgb = None
                tile_detail_score = None
            elif args.blend == "max":
                tile_image = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                tile_valid = np.zeros((tile_h, tile_w), dtype=np.uint8)
                tile_accum_rgb = None
                tile_accum_weight = None
                tile_max_score = np.full((tile_h, tile_w), -1, dtype=np.int16)
                tile_best_score = None
                tile_detail_rgb = None
                tile_detail_score = None
            elif args.blend == "best":
                tile_image = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                tile_valid = np.zeros((tile_h, tile_w), dtype=np.uint8)
                tile_accum_rgb = None
                tile_accum_weight = None
                tile_max_score = None
                tile_best_score = np.full((tile_h, tile_w), -1.0, dtype=np.float32)
                tile_detail_rgb = None
                tile_detail_score = None
            elif args.blend == "smart":
                tile_accum_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
                tile_accum_weight = np.zeros((tile_h, tile_w), dtype=np.float32)
                tile_image = None
                tile_valid = np.zeros((tile_h, tile_w), dtype=np.uint8)
                tile_max_score = None
                tile_best_score = None
                tile_detail_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                tile_detail_score = np.zeros((tile_h, tile_w), dtype=np.float32)
            else:
                tile_accum_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
                tile_accum_weight = np.zeros((tile_h, tile_w), dtype=np.float32)
                tile_image = None
                tile_valid = np.zeros((tile_h, tile_w), dtype=np.uint8)
                tile_max_score = None
                tile_best_score = None
                tile_detail_rgb = None
                tile_detail_score = None

            T_world_to_tile = np.array(
                [
                    [1.0, 0.0, T_world_to_canvas[0, 2] - tile_x0],
                    [0.0, 1.0, T_world_to_canvas[1, 2] - tile_y0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            for pose in relevant_poses:
                image = load_cached_frame(frame_paths[pose.frame_index])
                H_frame_to_tile = T_world_to_tile @ pose.H_frame_to_world
                warped = cv2.warpPerspective(image, H_frame_to_tile, (tile_w, tile_h))
                mask = cv2.warpPerspective(
                    np.full(image.shape[:2], 255, dtype=np.uint8),
                    H_frame_to_tile,
                    (tile_w, tile_h),
                    flags=cv2.INTER_NEAREST,
                )
                mask_bool = mask > 0
                if args.blend == "last":
                    tile_image[mask_bool] = warped[mask_bool]
                    tile_valid[mask_bool] = 255
                elif args.blend == "max":
                    warped_score = compute_heat_score(warped).astype(np.int16)
                    update_mask = mask_bool & (warped_score >= tile_max_score)
                    tile_image[update_mask] = warped[update_mask]
                    tile_max_score[update_mask] = warped_score[update_mask]
                    tile_valid[mask_bool] = 255
                elif args.blend == "best":
                    quality_gain = smart_weight_gain(frame_analysis, pose.frame_index)
                    score = cv2.warpPerspective(
                        compute_selection_score_map(image, quality_gain),
                        H_frame_to_tile,
                        (tile_w, tile_h),
                        flags=cv2.INTER_LINEAR,
                    )
                    score *= mask_bool.astype(np.float32)
                    update_mask = mask_bool & (score >= tile_best_score)
                    tile_image[update_mask] = warped[update_mask]
                    tile_best_score[update_mask] = score[update_mask]
                    tile_valid[mask_bool] = 255
                elif args.blend == "smart":
                    quality_gain = smart_weight_gain(frame_analysis, pose.frame_index)
                    weight = cv2.warpPerspective(
                        frame_weight_map((image.shape[1], image.shape[0]), "feather"),
                        H_frame_to_tile,
                        (tile_w, tile_h),
                    )
                    detail = cv2.warpPerspective(
                        compute_detail_score_map(image),
                        H_frame_to_tile,
                        (tile_w, tile_h),
                    )
                    weight *= mask_bool.astype(np.float32)
                    detail *= mask_bool.astype(np.float32)
                    weight *= quality_gain
                    tile_accum_rgb += warped.astype(np.float32) * weight[:, :, None]
                    tile_accum_weight += weight
                    detail_candidate = np.clip(
                        detail * np.clip(weight, 0.25, 1.0) * quality_gain,
                        0.0,
                        1.0,
                    )
                    update_best_detail_layer(
                        tile_detail_rgb,
                        tile_detail_score,
                        warped,
                        detail_candidate,
                        mask_bool,
                    )
                    tile_valid[tile_accum_weight > 0] = 255
                else:
                    weight = cv2.warpPerspective(
                        frame_weight_map((image.shape[1], image.shape[0]), args.blend),
                        H_frame_to_tile,
                        (tile_w, tile_h),
                    )
                    weight *= mask_bool.astype(np.float32)
                    tile_accum_rgb += warped.astype(np.float32) * weight[:, :, None]
                    tile_accum_weight += weight
                    tile_valid[tile_accum_weight > 0] = 255

            if args.blend == "smart":
                tile_image = compose_smart_blend(
                    tile_accum_rgb,
                    tile_accum_weight,
                    tile_detail_rgb,
                    tile_detail_score,
                )
            elif args.blend not in {"last", "max", "best"}:
                safe_weight = np.where(tile_accum_weight > 1e-6, tile_accum_weight, 1.0)
                tile_image = np.clip(
                    tile_accum_rgb / safe_weight[:, :, None],
                    0,
                    255,
                ).astype(np.uint8)

            if not np.any(tile_valid):
                continue

            tile_name = f"tile_y{tile_y0:06d}_x{tile_x0:06d}.png"
            tile_path = tile_dir / tile_name
            if not cv2.imwrite(str(tile_path), tile_image):
                raise RuntimeError(f"Failed to write tile: {tile_path}")
            manifest_tiles.append(
                {
                    "file": tile_name,
                    "x": tile_x0,
                    "y": tile_y0,
                    "width": tile_w,
                    "height": tile_h,
                }
            )

    if args.preview:
        render_preview_canvas(
            accepted_poses,
            frame_paths,
            T_world_to_canvas,
            canvas_w,
            canvas_h,
            preview_blend,
            Path(args.preview).resolve(),
            frame_analysis,
        )

    manifest = {
        "original_video": str(video_path),
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "world_bounds": {
            "min_x": bounds[0],
            "min_y": bounds[1],
            "max_x": bounds[2],
            "max_y": bounds[3],
        },
        "margin_px": args.margin_px,
        "tile_size": args.tile_size,
        "accepted_frame_count": len(accepted_poses),
        "args": vars(args),
        "tiles": manifest_tiles,
    }
    with (tile_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def draw_trajectory_image(
    accepted_poses: Sequence[FramePose],
    bounds: tuple[float, float, float, float],
    output_path: Path,
    margin_px: int,
) -> None:
    T_world_to_canvas, canvas_w, canvas_h = make_world_to_canvas_transform(bounds, margin_px)
    scale = min(1.0, 1600.0 / max(canvas_w, canvas_h))
    canvas_w = max(1, int(math.ceil(canvas_w * scale)))
    canvas_h = max(1, int(math.ceil(canvas_h * scale)))
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    T = S @ T_world_to_canvas
    image = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    centers = np.array(
        [pose.center_world for pose in accepted_poses if pose.center_world is not None],
        dtype=np.float64,
    )
    if len(centers) >= 2:
        centers_canvas = transform_canvas_points(centers, T).astype(np.int32)
        cv2.polylines(image, [centers_canvas], False, (32, 80, 220), 2, cv2.LINE_AA)
    for pose in accepted_poses[:: max(1, len(accepted_poses) // 20)]:
        if pose.corners_world is None:
            continue
        polygon = transform_canvas_points(pose.corners_world, T).astype(np.int32)
        cv2.polylines(image, [polygon], True, (0, 180, 0), 1, cv2.LINE_AA)
    ensure_parent_dir(output_path)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write trajectory image: {output_path}")


def save_mosaic(output_path: Path, render_result: RenderResult) -> None:
    if render_result.mosaic_bgr is None:
        raise RuntimeError("No mosaic image was rendered.")
    ensure_parent_dir(output_path)
    if not cv2.imwrite(str(output_path), render_result.mosaic_bgr):
        raise RuntimeError(f"Failed to write mosaic image: {output_path}")


def inspect_output_limits(
    canvas_w: int,
    canvas_h: int,
    args: argparse.Namespace,
) -> None:
    megapixels = (canvas_w * canvas_h) / 1_000_000.0
    estimated_bytes = estimate_canvas_memory_bytes(canvas_w, canvas_h, args.blend)
    print(
        f"Canvas size: {canvas_w}x{canvas_h} ({megapixels:.2f} MP), "
        f"estimated render memory {estimated_bytes / (1024 ** 3):.2f} GiB"
    )
    if megapixels > args.max_canvas_mp and not args.enable_tiling:
        raise RuntimeError(
            "Computed canvas exceeds --max-canvas-mp. "
            f"Canvas is {megapixels:.2f} MP, limit is {args.max_canvas_mp:.2f} MP. "
            "Use --enable-tiling, reduce --max-dim, or increase --min-step-px."
        )
    if args.video_view == "canvas" and (
        args.output_video or args.comparison_video or args.comparison_edge_video
    ):
        if megapixels > min(args.max_canvas_mp, 32.0):
            raise RuntimeError(
                "--video-view canvas is only practical for modest canvases. "
                "Use --video-view overview or --video-view follow."
            )


def main() -> int:
    args = parse_args()
    crop = parse_crop(args.crop)
    video_path = open_video_or_autodetect(args.input)
    video_info = get_video_info(video_path)
    if args.video_fps <= 0:
        args.video_fps = float(video_info["fps"]) if video_info["fps"] else 20.0
    if (args.output_video or args.comparison_video or args.comparison_edge_video) and not any(
        (
            args.draw_trajectory,
            args.draw_current_frame_outline,
            args.draw_frame_index,
            args.draw_quality,
        )
    ):
        args.draw_trajectory = True
        args.draw_current_frame_outline = True
        args.draw_frame_index = True
        args.draw_quality = True
    print(
        "Input video info: "
        f"{video_info['width']}x{video_info['height']} px, "
        f"{video_info['fps']:.2f} fps, "
        f"{video_info['frame_count']} frames"
    )

    all_poses, accepted_poses, processed_shape = compose_global_poses(video_path, crop, args)
    if args.presentation_align_motion_right:
        apply_presentation_alignment(accepted_poses)
    accepted_count = len(accepted_poses)
    skipped_count = len(all_poses) - accepted_count
    print(f"Processed sampled frames: {len(all_poses)}")
    print(f"Accepted frames: {accepted_count}, skipped or rejected: {skipped_count}")
    print(f"Processed frame size: {processed_shape[0]}x{processed_shape[1]}")

    bounds = compute_world_bounds(accepted_poses)
    print(
        "World bounds: "
        f"min=({bounds[0]:.1f}, {bounds[1]:.1f}) "
        f"max=({bounds[2]:.1f}, {bounds[3]:.1f})"
    )
    T_world_to_canvas, canvas_w, canvas_h = make_world_to_canvas_transform(bounds, args.margin_px)
    inspect_output_limits(canvas_w, canvas_h, args)

    if args.write_poses:
        csv_path = Path(args.write_poses).resolve()
        write_pose_log_csv(csv_path, all_poses)
        print(f"Wrote pose CSV: {csv_path}")
    if args.write_json:
        json_path = Path(args.write_json).resolve()
        write_pose_log_json(json_path, all_poses)
        print(f"Wrote pose JSON: {json_path}")
    if args.trajectory:
        trajectory_path = Path(args.trajectory).resolve()
        draw_trajectory_image(accepted_poses, bounds, trajectory_path, args.margin_px)
        print(f"Wrote trajectory image: {trajectory_path}")

    cache_dir_ctx, frame_paths, raw_frame_paths = cache_accepted_frames(
        video_path,
        accepted_poses,
        crop,
        args,
    )
    print(f"Cached {len(frame_paths)} accepted frames under {cache_dir_ctx.name}")
    preview_blend = resolve_preview_blend(args)
    needs_content_analysis = args.blend in {"smart", "best"} or preview_blend in {"smart", "best"}
    frame_analysis = (
        analyze_accepted_frame_content(accepted_poses, frame_paths)
        if needs_content_analysis
        else None
    )
    if frame_analysis:
        print(f"Analyzed {len(frame_analysis)} accepted frames for global content ranking")

    try:
        if args.enable_tiling or (canvas_w * canvas_h) / 1_000_000.0 > args.max_canvas_mp:
            if args.comparison_video or args.comparison_edge_video:
                raise RuntimeError(
                    "Comparison video rendering currently requires single-canvas rendering. "
                    "Reduce the canvas size or disable tiling for this run."
                )
            render_mosaic_tiles(
                accepted_poses,
                frame_paths,
                frame_analysis,
                video_path,
                bounds,
                T_world_to_canvas,
                canvas_w,
                canvas_h,
                args,
            )
            print(f"Wrote tile set: {Path(args.tile_output_dir).resolve()}")
            if args.output_video:
                print(
                    "Warning: progress video is only rendered in single-canvas mode in this version; "
                    "tile output was generated without a progress video."
                )
        else:
            render_result = render_mosaic_single_canvas(
                accepted_poses,
                frame_paths,
                raw_frame_paths,
                frame_analysis,
                bounds,
                T_world_to_canvas,
                canvas_w,
                canvas_h,
                args,
            )
            output_path = Path(args.output).resolve()
            save_mosaic(output_path, render_result)
            print(f"Wrote mosaic image: {output_path}")
            if args.preview:
                preview_path = Path(args.preview).resolve()
                render_preview_canvas(
                    accepted_poses,
                    frame_paths,
                    T_world_to_canvas,
                    canvas_w,
                    canvas_h,
                    preview_blend,
                    preview_path,
                    frame_analysis,
                )
                print(f"Wrote preview image: {preview_path}")
            if args.output_video:
                print(f"Wrote progress video: {Path(args.output_video).resolve()}")
            if args.comparison_video:
                print(f"Wrote comparison video: {Path(args.comparison_video).resolve()}")
            if args.comparison_edge_video:
                print(f"Wrote edge comparison video: {Path(args.comparison_edge_video).resolve()}")
    finally:
        cache_dir_ctx.cleanup()

    rejected_frames = [pose for pose in all_poses if pose.H_frame_to_world is None]
    if rejected_frames:
        print(
            f"Transform failures: {len(rejected_frames)}. "
            "If alignment is unstable, try --every 1, --model partial-affine, "
            "--min-matches 12, --min-inliers 8, or add --crop."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
