#!/usr/bin/env python3
"""Live webcam viewer for the floor/heat-map stitcher."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from floor_video_map import (
    DEFAULT_VIDEO_CODEC,
    ProcessedFrame,
    clamp_memory_alpha,
    compute_frame_geometry,
    crop_to_valid_region,
    detect_and_describe,
    ensure_parent_dir,
    estimate_pair_transform,
    estimate_rotation_deg_from_H,
    make_detector,
    make_matcher,
    normalize_angle_deg,
    parse_crop,
    preprocess_frame,
    put_text_lines,
    scale_and_letterbox,
    transform_points,
    update_memory_canvas,
    validate_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a live camera feed and a real-time memory mosaic side by side."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or stream URL. Default: 0.",
    )
    parser.add_argument("--output", default="live_mosaic.png", help="Mosaic image saved on exit.")
    parser.add_argument("--record", help="Optional side-by-side preview video output.")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV windows.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N captured frames.")
    parser.add_argument("--process-every", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--capture-width", type=int, default=0, help="Requested camera width.")
    parser.add_argument("--capture-height", type=int, default=0, help="Requested camera height.")
    parser.add_argument("--mirror", action="store_true", help="Mirror camera frames horizontally.")
    parser.add_argument("--flip-vertical", action="store_true", help="Flip camera frames vertically.")
    parser.add_argument("--crop", help="Crop processed frames as x,y,w,h before resizing.")
    parser.add_argument("--max-dim", type=int, default=640)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--canvas-width", type=int, default=2200)
    parser.add_argument("--canvas-height", type=int, default=1600)
    parser.add_argument("--canvas-pad", type=int, default=120)
    parser.add_argument("--max-canvas-mp", type=float, default=64.0)
    parser.add_argument("--detector", choices=("sift", "orb"), default="sift")
    parser.add_argument("--nfeatures", type=int, default=2500)
    parser.add_argument(
        "--model",
        choices=("translation", "partial-affine", "affine", "homography"),
        default="partial-affine",
    )
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac", type=float, default=3.0)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.2)
    parser.add_argument("--memory-alpha", type=float, default=0.45)
    parser.add_argument("--anchor-step-px", type=float, default=8.0)
    parser.add_argument("--anchor-rotation-deg", type=float, default=1.0)
    parser.add_argument(
        "--no-lock-small-motion-updates",
        action="store_false",
        dest="lock_small_motion_updates",
        help="Paint every tracked pose directly instead of locking tiny updates to the anchor.",
    )
    parser.set_defaults(lock_small_motion_updates=True)
    parser.add_argument("--use-ecc", action="store_true")
    parser.add_argument("--disable-flow", action="store_true")
    parser.add_argument("--flow-max-points", type=int, default=300)
    parser.add_argument("--flow-min-distance", type=float, default=12.0)
    parser.add_argument("--flow-quality", type=float, default=0.01)
    parser.add_argument("--flow-error-threshold", type=float, default=20.0)
    parser.add_argument("--flow-backcheck-threshold", type=float, default=1.5)
    parser.add_argument("--min-ecc", type=float, default=0.85)
    parser.add_argument("--max-translation-factor", type=float, default=1.5)
    parser.add_argument("--max-rotation-jump-deg", type=float, default=75.0)
    parser.add_argument("--min-scale", type=float, default=0.6)
    parser.add_argument("--max-scale", type=float, default=1.5)
    parser.add_argument("--min-determinant", type=float, default=0.2)
    parser.add_argument("--max-determinant", type=float, default=5.0)
    parser.add_argument(
        "--reacquire-after",
        type=int,
        default=12,
        help="Use the current frame as a fresh local reference after this many rejected frames.",
    )
    return parser.parse_args()


def build_tracking_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        ratio=args.ratio,
        ransac=args.ransac,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        use_ecc=args.use_ecc,
        disable_flow=args.disable_flow,
        flow_max_points=args.flow_max_points,
        flow_min_distance=args.flow_min_distance,
        flow_quality=args.flow_quality,
        flow_error_threshold=args.flow_error_threshold,
        flow_backcheck_threshold=args.flow_backcheck_threshold,
        min_ecc=args.min_ecc,
        max_translation_factor=args.max_translation_factor,
        max_rotation_jump_deg=args.max_rotation_jump_deg,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        min_determinant=args.min_determinant,
        max_determinant=args.max_determinant,
    )


def parse_video_source(source_text: str) -> int | str:
    stripped = source_text.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source = parse_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if args.capture_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    if args.capture_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera/source: {args.source}")
    return cap


def make_canvas_transform(canvas_shape: tuple[int, int], frame_size: tuple[int, int]) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    frame_w, frame_h = frame_size
    offset_x = canvas_w * 0.5 - frame_w * 0.5
    offset_y = canvas_h * 0.5 - frame_h * 0.5
    return np.array(
        [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class LiveWebcamMapper:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.crop = parse_crop(args.crop)
        self.tracking_args = build_tracking_args(args)
        self.detector, detector_name = make_detector(args.detector, args.nfeatures)
        self.matcher = make_matcher(detector_name)
        self.memory_alpha = clamp_memory_alpha(args.memory_alpha)
        self.reset()

    def reset(self) -> None:
        self.canvas = np.zeros(
            (self.args.canvas_height, self.args.canvas_width, 3),
            dtype=np.uint8,
        )
        self.valid_mask = np.zeros((self.args.canvas_height, self.args.canvas_width), dtype=np.uint8)
        self.T_world_to_canvas: np.ndarray | None = None
        self.previous_frame: ProcessedFrame | None = None
        self.anchor_H: np.ndarray | None = None
        self.anchor_center: tuple[float, float] | None = None
        self.frame_size: tuple[int, int] | None = None
        self.frame_index = 0
        self.mapped_frames = 0
        self.tracked_frames = 0
        self.rejected_frames = 0
        self.lost_streak = 0
        self.status = "waiting for frames"
        self.last_source_view: np.ndarray | None = None

    def maybe_expand_canvas(self, corners_canvas: np.ndarray) -> bool:
        pad = self.args.canvas_pad
        min_x = float(np.min(corners_canvas[:, 0]))
        min_y = float(np.min(corners_canvas[:, 1]))
        max_x = float(np.max(corners_canvas[:, 0]))
        max_y = float(np.max(corners_canvas[:, 1]))
        canvas_h, canvas_w = self.canvas.shape[:2]

        left = max(0, int(np.ceil(pad - min_x)))
        top = max(0, int(np.ceil(pad - min_y)))
        right = max(0, int(np.ceil(max_x + pad - canvas_w)))
        bottom = max(0, int(np.ceil(max_y + pad - canvas_h)))
        if not any((left, top, right, bottom)):
            return False

        new_w = canvas_w + left + right
        new_h = canvas_h + top + bottom
        if (new_w * new_h) / 1_000_000.0 > self.args.max_canvas_mp:
            self.status = "canvas limit reached; increase --max-canvas-mp or reset"
            return False

        expanded_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        expanded_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        expanded_canvas[top : top + canvas_h, left : left + canvas_w] = self.canvas
        expanded_mask[top : top + canvas_h, left : left + canvas_w] = self.valid_mask
        self.canvas = expanded_canvas
        self.valid_mask = expanded_mask
        assert self.T_world_to_canvas is not None
        self.T_world_to_canvas[0, 2] += left
        self.T_world_to_canvas[1, 2] += top
        return True

    def paint(self, image_bgr: np.ndarray, H_frame_to_world: np.ndarray) -> None:
        if self.T_world_to_canvas is None:
            self.T_world_to_canvas = make_canvas_transform(
                self.canvas.shape[:2],
                (image_bgr.shape[1], image_bgr.shape[0]),
            )

        local_corners = np.array(
            [
                [0.0, 0.0],
                [float(image_bgr.shape[1]), 0.0],
                [float(image_bgr.shape[1]), float(image_bgr.shape[0])],
                [0.0, float(image_bgr.shape[0])],
            ],
            dtype=np.float64,
        )
        for _ in range(2):
            H_frame_to_canvas = self.T_world_to_canvas @ H_frame_to_world
            corners_canvas = transform_points(H_frame_to_canvas, local_corners)
            if not self.maybe_expand_canvas(corners_canvas):
                break

        H_frame_to_canvas = self.T_world_to_canvas @ H_frame_to_world
        canvas_h, canvas_w = self.canvas.shape[:2]
        warped = cv2.warpPerspective(
            image_bgr,
            H_frame_to_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpPerspective(
            np.full(image_bgr.shape[:2], 255, dtype=np.uint8),
            H_frame_to_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        update_memory_canvas(self.canvas, self.valid_mask, warped, mask > 0, self.memory_alpha)
        self.mapped_frames += 1

    def paint_pose_for_tracking(self, color: np.ndarray, H_frame_to_world: np.ndarray) -> str:
        H_to_paint = H_frame_to_world
        paint_mode = "raw"
        if self.args.lock_small_motion_updates and self.anchor_H is not None and self.anchor_center is not None:
            current_center, _ = compute_frame_geometry(H_frame_to_world, color.shape[1], color.shape[0])
            step = float(np.linalg.norm(np.asarray(current_center) - np.asarray(self.anchor_center)))
            rotation_delta = abs(
                normalize_angle_deg(
                    estimate_rotation_deg_from_H(H_frame_to_world)
                    - estimate_rotation_deg_from_H(self.anchor_H)
                )
            )
            if step < self.args.anchor_step_px and rotation_delta < self.args.anchor_rotation_deg:
                H_to_paint = self.anchor_H
                paint_mode = "locked"
            else:
                self.anchor_H = H_frame_to_world.copy()
                self.anchor_center = current_center
                paint_mode = "anchored"
        else:
            self.anchor_H = H_frame_to_world.copy()
            self.anchor_center, _ = compute_frame_geometry(H_frame_to_world, color.shape[1], color.shape[0])
        self.paint(color, H_to_paint)
        return paint_mode

    def process(self, frame_bgr: np.ndarray, timestamp_sec: float) -> np.ndarray:
        color, gray = preprocess_frame(frame_bgr, self.crop, self.args.max_dim)
        self.last_source_view = color
        self.frame_size = (color.shape[1], color.shape[0])
        keypoints, descriptors = detect_and_describe(gray, self.detector)

        if self.previous_frame is None:
            H_identity = np.eye(3, dtype=np.float64)
            self.previous_frame = ProcessedFrame(
                frame_index=self.frame_index,
                timestamp_sec=timestamp_sec,
                color=color,
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                H_frame_to_world=H_identity,
                tracking_mode="bootstrap",
            )
            self.anchor_H = H_identity.copy()
            self.anchor_center, _ = compute_frame_geometry(H_identity, color.shape[1], color.shape[0])
            self.paint(color, H_identity)
            self.status = f"initialized  keypoints={len(keypoints)}"
            return color

        estimate = estimate_pair_transform(
            self.previous_frame,
            gray,
            keypoints,
            descriptors,
            self.matcher,
            self.tracking_args,
        )
        verdict = (
            (False, estimate.reason)
            if estimate.H_current_to_reference is None
            else validate_transform(
                estimate.H_current_to_reference,
                estimate,
                color.shape,
                self.tracking_args,
            )
        )

        if verdict[0]:
            assert estimate.H_current_to_reference is not None
            H_frame_to_world = self.previous_frame.H_frame_to_world @ estimate.H_current_to_reference
            paint_mode = self.paint_pose_for_tracking(color, H_frame_to_world)
            self.previous_frame = ProcessedFrame(
                frame_index=self.frame_index,
                timestamp_sec=timestamp_sec,
                color=color,
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                H_frame_to_world=H_frame_to_world,
                tracking_mode="visual",
            )
            self.tracked_frames += 1
            self.lost_streak = 0
            source = "ECC" if estimate.used_ecc else "flow" if estimate.used_flow else "features"
            self.status = (
                f"{source} ok  inliers={estimate.num_inliers}/{estimate.num_matches}  "
                f"ratio={estimate.inlier_ratio:.2f}  paint={paint_mode}"
            )
            return color

        self.rejected_frames += 1
        self.lost_streak += 1
        self.status = f"tracking rejected: {verdict[1]}"
        if self.lost_streak >= self.args.reacquire_after:
            H_reference = self.anchor_H.copy() if self.anchor_H is not None else np.eye(3, dtype=np.float64)
            self.previous_frame = ProcessedFrame(
                frame_index=self.frame_index,
                timestamp_sec=timestamp_sec,
                color=color,
                gray=gray,
                keypoints=keypoints,
                descriptors=descriptors,
                H_frame_to_world=H_reference,
                tracking_mode="reacquire",
            )
            self.lost_streak = 0
            self.status = "reacquiring from current frame"
        return color

    def display_frame(self, source_bgr: np.ndarray) -> np.ndarray:
        panel_w = self.args.display_width // 2
        panel_h = self.args.display_height
        source_panel = scale_and_letterbox(source_bgr, panel_w, panel_h)

        if np.any(self.valid_mask):
            map_crop, _, _ = crop_to_valid_region(self.canvas, self.valid_mask)
        else:
            map_crop = self.canvas
        map_panel = scale_and_letterbox(map_crop, self.args.display_width - panel_w, panel_h)

        put_text_lines(source_panel, ["Camera", f"frame {self.frame_index}"], origin_xy=(12, 26))
        put_text_lines(
            map_panel,
            [
                "Live memory map",
                f"mapped {self.mapped_frames}  tracked {self.tracked_frames}  rejected {self.rejected_frames}",
                self.status,
            ],
            origin_xy=(12, 26),
        )
        return np.hstack([source_panel, map_panel])

    def save_output(self, output_path: str | None) -> None:
        if not output_path:
            return
        if np.any(self.valid_mask):
            mosaic, _, _ = crop_to_valid_region(self.canvas, self.valid_mask)
        else:
            mosaic = self.canvas
        path = Path(output_path).resolve()
        ensure_parent_dir(path)
        if not cv2.imwrite(str(path), mosaic):
            raise RuntimeError(f"Failed to write live mosaic: {path}")
        print(f"Wrote live mosaic: {path}")


def normalize_frame(frame_bgr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.mirror:
        frame_bgr = cv2.flip(frame_bgr, 1)
    if args.flip_vertical:
        frame_bgr = cv2.flip(frame_bgr, 0)
    return frame_bgr


def main() -> int:
    args = parse_args()
    if args.process_every <= 0:
        raise ValueError("--process-every must be positive")
    mapper = LiveWebcamMapper(args)
    cap = open_capture(args)

    writer: cv2.VideoWriter | None = None
    if args.record:
        record_path = Path(args.record).resolve()
        ensure_parent_dir(record_path)
        fourcc = cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(
            str(record_path),
            fourcc,
            float(fps),
            (args.display_width, args.display_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open preview recorder: {record_path}")

    print("Live webcam map is running.")
    print("Keys: q/Esc quit, r reset map, s save current mosaic.")
    start_time = time.monotonic()
    last_display = np.zeros((args.display_height, args.display_width, 3), dtype=np.uint8)
    try:
        captured = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_bgr = normalize_frame(frame_bgr, args)
            timestamp_sec = time.monotonic() - start_time
            if captured % args.process_every == 0:
                source_view = mapper.process(frame_bgr, timestamp_sec)
            elif mapper.last_source_view is not None:
                source_view = mapper.last_source_view
            else:
                source_view, _ = preprocess_frame(frame_bgr, mapper.crop, args.max_dim)

            last_display = mapper.display_frame(source_view)
            if writer is not None:
                writer.write(last_display)
            if not args.headless:
                cv2.imshow("Live Webcam Floor Map", last_display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    mapper.reset()
                    print("Reset live mosaic.")
                elif key == ord("s"):
                    mapper.save_output(args.output)

            captured += 1
            mapper.frame_index += 1
            if args.max_frames > 0 and captured >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Wrote preview video: {Path(args.record).resolve()}")
        if not args.headless:
            cv2.destroyAllWindows()

    mapper.save_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
