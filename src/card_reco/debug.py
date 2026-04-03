"""Debug visualisation helpers for the card recognition pipeline.

When debug mode is active, intermediate images are written to an output
directory so a human can inspect each stage of detection, hashing, and
matching.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


class DebugWriter:
    """Write debug images to *output_dir* for a single pipeline run.

    Create one instance per ``identify`` invocation and pass it through the
    pipeline.  Calling code that does **not** want debug output simply passes
    ``None`` wherever a ``DebugWriter | None`` is accepted.
    """

    def __init__(self, output_dir: str | Path, *, clean: bool = True) -> None:
        self.output_dir = Path(output_dir)
        if clean and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _next_prefix(self, label: str) -> str:
        self._step += 1
        return f"{self._step:02d}_{label}"

    def save(self, name: str, image: np.ndarray) -> Path:
        """Write *image* to *output_dir/name*.  Returns the written path."""
        path = self.output_dir / name
        cv2.imwrite(str(path), image)
        return path

    # ------------------------------------------------------------------
    # Detection stage
    # ------------------------------------------------------------------

    def save_input(self, image: np.ndarray) -> None:
        """Save the original input image."""
        self.save(f"{self._next_prefix('input')}.png", image)

    def save_preprocessed(
        self,
        gray: np.ndarray,
        blurred: np.ndarray,
    ) -> None:
        """Save grayscale and blurred preprocessing results."""
        prefix = self._next_prefix("preprocess")
        self.save(f"{prefix}_gray.png", gray)
        self.save(f"{prefix}_blurred.png", blurred)

    def save_edge_map(
        self,
        label: str,
        edge_image: np.ndarray,
    ) -> None:
        """Save an edge / threshold / mask image from a detection strategy."""
        self.save(f"{self._next_prefix(label)}.png", edge_image)

    def save_candidates(
        self,
        image: np.ndarray,
        contours: list,
        min_area: float,
    ) -> None:
        """Draw all candidate contours on the original image."""
        vis = image.copy()
        image_area = image.shape[0] * image.shape[1]
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            pct = area / image_area * 100
            # Color: green for larger, blue for smaller
            color = (0, 255, 0) if area > min_area * 5 else (255, 180, 0)
            cv2.drawContours(vis, [cnt], -1, color, 2)
            moments = cv2.moments(cnt)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                cv2.putText(
                    vis,
                    f"#{i} {pct:.1f}%",
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        self.save(f"{self._next_prefix('candidates')}.png", vis)

    def save_corners(
        self,
        image: np.ndarray,
        corners_list: list[np.ndarray],
        labels: list[str] | None = None,
    ) -> None:
        """Draw detected corner quadrilaterals on the original image."""
        vis = image.copy()
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for i, corners in enumerate(corners_list):
            color = colors[i % len(colors)]
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, color, 3)
            # Number each corner
            for j, pt in enumerate(corners.astype(np.int32)):
                cv2.circle(vis, tuple(pt), 6, color, -1)
                cv2.putText(
                    vis,
                    str(j),
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            # Label
            label = labels[i] if labels else f"det{i}"
            tl = corners[0].astype(int)
            cv2.putText(
                vis,
                label,
                (tl[0], tl[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        self.save(f"{self._next_prefix('corners')}.png", vis)

    def save_nms_result(
        self,
        image: np.ndarray,
        before_count: int,
        after_detections: list,
    ) -> None:
        """Annotate NMS survivors on the original image."""
        vis = image.copy()
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]
        for i, det in enumerate(after_detections):
            color = colors[i % len(colors)]
            pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, color, 3)
            area = cv2.contourArea(pts)
            image_area = image.shape[0] * image.shape[1]
            pct = area / image_area * 100
            tl = det.corners[0].astype(int)
            cv2.putText(
                vis,
                f"det{i} {pct:.1f}% conf={det.confidence:.2f}",
                (tl[0], tl[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.putText(
            vis,
            f"NMS: {before_count} -> {len(after_detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        self.save(f"{self._next_prefix('nms')}.png", vis)

    def save_warped(
        self,
        index: int,
        warped: np.ndarray,
    ) -> None:
        """Save a single warped (perspective-corrected) card image."""
        self.save(f"{self._next_prefix('warped')}_{index}.png", warped)

    # ------------------------------------------------------------------
    # Matching stage
    # ------------------------------------------------------------------

    def save_match_summary(
        self,
        index: int,
        card_image: np.ndarray,
        matches: list,
    ) -> None:
        """Save warped card side-by-side with a text summary of matches."""
        # Build a canvas: card on left, text on right
        h, w = card_image.shape[:2]
        text_w = max(500, w)
        canvas = np.full((h, w + text_w, 3), 255, dtype=np.uint8)
        canvas[:h, :w] = card_image

        y = 30
        cv2.putText(
            canvas,
            f"Card {index} matches:",
            (w + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        y += 35
        for match in matches[:5]:
            text = (
                f"#{match.rank} {match.card.name} "
                f"({match.card.set_id}-{match.card.number}) "
                f"d={match.distance:.1f}"
            )
            cv2.putText(
                canvas,
                text,
                (w + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            y += 25

            # Per-hash distances
            if match.distances:
                parts = " ".join(f"{k}={v}" for k, v in match.distances.items())
                cv2.putText(
                    canvas,
                    f"   {parts}",
                    (w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (100, 100, 100),
                    1,
                )
                y += 22

        if not matches:
            cv2.putText(
                canvas,
                "No matches found",
                (w + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 200),
                2,
            )

        self.save(f"{self._next_prefix('match')}_{index}.png", canvas)
