"""Live scanner GUI -- capture screen, identify cards in real time."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from card_reco.detector import detect_cards
from card_reco.detector.nms import compute_overlap
from card_reco.pipeline import identify_detections

if TYPE_CHECKING:
    from card_reco.embedder import CardEmbedder
    from card_reco.faiss_index import CardIndex
    from card_reco.models import DetectedCard, MatchResult

# How often the live preview refreshes (milliseconds).
_PREVIEW_INTERVAL_MS = 100

# Maximum dimensions for preview and result panels.
_PANEL_MAX_W = 640
_PANEL_MAX_H = 480

# Shared colour palette for detection overlays.
_OVERLAY_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

# --- Tracking constants ---
# IoU threshold to match a new detection to an existing tracked card.
_IOU_MATCH_THRESH = 0.3
# Remove a tracked card after this many consecutive missed frames.
_MAX_MISSED_FRAMES = 5
# Number of recent identification votes to consider.
_VOTE_WINDOW = 10


def _scale_image(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize *image* to fit within *max_w* x *max_h*, preserving aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_w and h <= max_h:
        return image
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return np.asarray(
        cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )


def _bgr_to_photoimage(image: np.ndarray) -> ImageTk.PhotoImage:
    """Convert a BGR numpy array to a tkinter-compatible PhotoImage."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def _draw_detections(
    image: np.ndarray,
    detections: list[DetectedCard],
    results: list[list[MatchResult]],
) -> np.ndarray:
    """Draw detection overlays (corners + card names) on *image*."""
    vis = image.copy()
    for i, det in enumerate(detections):
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2)

        # Show top-match name near the top-left corner.
        if i < len(results) and results[i]:
            label = results[i][0].card.name
            dist = results[i][0].distance
            text = f"{label} ({dist:.2f})"
        else:
            text = "No match"
        tl = det.corners[0].astype(int)
        cv2.putText(
            vis,
            text,
            (tl[0], max(tl[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return vis


# ──────────────────────────────────────────────────────────────────
#  Card tracker — temporal voting across frames
# ──────────────────────────────────────────────────────────────────


@dataclass
class _TrackedCard:
    """A card being tracked across consecutive frames."""

    card_id: int
    corners: np.ndarray
    matches: list[MatchResult]
    vote_history: list[str] = field(default_factory=list)
    best_per_id: dict[str, MatchResult] = field(default_factory=dict)
    frames_seen: int = 1
    frames_missed: int = 0

    @property
    def voted_match(self) -> MatchResult | None:
        """Return the ``MatchResult`` for the most-voted card ID."""
        if not self.vote_history:
            return None
        best_id = Counter(self.vote_history).most_common(1)[0][0]
        return self.best_per_id.get(best_id)

    @property
    def vote_confidence(self) -> float:
        """Fraction of votes for the top card (0-1)."""
        if not self.vote_history:
            return 0.0
        top_count = Counter(self.vote_history).most_common(1)[0][1]
        return top_count / len(self.vote_history)


class CardTracker:
    """Tracks cards across frames using IoU matching and temporal voting.

    Each call to :meth:`update` matches the new frame's detections to
    existing tracked cards by IoU overlap.  Matched cards accumulate
    identification votes (top-1 card ID per frame).  Unmatched detections
    create new tracks; tracks that aren't seen for several frames are
    removed.

    Use :meth:`get_display_data` to retrieve stable, voted-best
    identification results for overlay rendering.
    """

    def __init__(self) -> None:
        self._tracked: list[_TrackedCard] = []
        self._next_id = 0

    # -------------------------------------------------------------- update

    def update(
        self,
        pairs: list[tuple[DetectedCard, list[MatchResult]]],
    ) -> None:
        """Incorporate a new frame's detection + identification results.

        *pairs* is a list of ``(DetectedCard, matches)`` tuples as
        returned by :func:`~card_reco.pipeline.identify_detections`.
        """
        matched_tracks: set[int] = set()
        matched_pairs: set[int] = set()

        # Build (iou, pair_idx, track_idx) triples, sort by IoU desc.
        iou_triples: list[tuple[float, int, int]] = []
        for p_idx, (det, _matches) in enumerate(pairs):
            for t_idx, track in enumerate(self._tracked):
                iou = compute_overlap(det.corners, track.corners)
                if iou >= _IOU_MATCH_THRESH:
                    iou_triples.append((iou, p_idx, t_idx))

        iou_triples.sort(reverse=True)

        for _iou, p_idx, t_idx in iou_triples:
            if p_idx in matched_pairs or t_idx in matched_tracks:
                continue
            matched_pairs.add(p_idx)
            matched_tracks.add(t_idx)
            det, matches = pairs[p_idx]
            self._update_track(self._tracked[t_idx], det.corners, matches)

        # New detections → new tracked cards.
        for p_idx, (det, matches) in enumerate(pairs):
            if p_idx in matched_pairs:
                continue
            self._add_track(det.corners, matches)

        # Increment missed frames for unmatched tracks.
        for t_idx, _track in enumerate(self._tracked):
            if t_idx not in matched_tracks:
                self._tracked[t_idx].frames_missed += 1

        # Prune stale tracks.
        self._tracked = [
            t for t in self._tracked if t.frames_missed <= _MAX_MISSED_FRAMES
        ]

    # --------------------------------------------------------- display data

    def get_display_data(
        self,
    ) -> list[tuple[np.ndarray, list[MatchResult]]]:
        """Return ``(corners, voted_matches)`` for each tracked card.

        The voted match list contains a single ``MatchResult`` — the
        one with the most votes across recent frames.  Falls back to
        the latest frame's matches if voting hasn't converged yet.
        """
        result: list[tuple[np.ndarray, list[MatchResult]]] = []
        for track in self._tracked:
            voted = track.voted_match
            if voted is not None:
                result.append((track.corners, [voted]))
            else:
                result.append((track.corners, track.matches))
        return result

    # --------------------------------------------------------------- clear

    def clear(self) -> None:
        """Remove all tracked cards."""
        self._tracked.clear()
        self._next_id = 0

    @property
    def count(self) -> int:
        """Number of currently tracked cards."""
        return len(self._tracked)

    # ------------------------------------------------------------ internals

    def _add_track(self, corners: np.ndarray, matches: list[MatchResult]) -> None:
        vote_history: list[str] = []
        best_per_id: dict[str, MatchResult] = {}
        if matches:
            card_id = matches[0].card.id
            vote_history.append(card_id)
            best_per_id[card_id] = matches[0]

        self._tracked.append(
            _TrackedCard(
                card_id=self._next_id,
                corners=corners,
                matches=matches,
                vote_history=vote_history,
                best_per_id=best_per_id,
            )
        )
        self._next_id += 1

    @staticmethod
    def _update_track(
        track: _TrackedCard,
        corners: np.ndarray,
        matches: list[MatchResult],
    ) -> None:
        track.corners = corners
        track.frames_seen += 1
        track.frames_missed = 0
        if matches:
            track.matches = matches
            card_id = matches[0].card.id
            track.vote_history.append(card_id)
            if len(track.vote_history) > _VOTE_WINDOW:
                track.vote_history = track.vote_history[-_VOTE_WINDOW:]
            existing = track.best_per_id.get(card_id)
            if existing is None or matches[0].distance > existing.distance:
                track.best_per_id[card_id] = matches[0]


# ──────────────────────────────────────────────────────────────────
#  Scanner GUI
# ──────────────────────────────────────────────────────────────────


class Scanner:
    """Tkinter GUI for real-time screen-capture card scanning.

    Supports two backends:

    * ``"cnn"`` (default) — CNN embedding + FAISS search.  Enables
      continuous real-time scanning with tracking and temporal voting.
    * ``"hash"`` — Perceptual hashing.  Single-shot scan only (legacy).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        monitor: int = 1,
        region: tuple[int, int, int, int] | None = None,
        top_n: int = 5,
        threshold: float = 40.0,
        backend: str = "cnn",
    ) -> None:
        self._db_path = db_path
        self._monitor_index = monitor
        self._region = region
        self._top_n = top_n
        self._threshold = threshold
        self._backend = backend

        # Screen-capture context (lazily created).
        self._sct: Any = None

        # CNN resources (created once in run()).
        self._embedder: CardEmbedder | None = None
        self._card_index: CardIndex | None = None

        # Card tracker for temporal voting.
        self._tracker = CardTracker()

        # Event signalling that CNN resources are loaded.
        self._cnn_ready = threading.Event()

        # Latest captured frame (BGR numpy array).  Written by the main-
        # thread preview loop, read by the background scan loop.
        self._current_frame: np.ndarray | None = None
        # Incremented each time a new frame is captured so the scan loop
        # can skip already-processed frames.
        self._frame_seq = 0

        # Scanning state.
        self._scanning = False
        self._scan_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Scan timing (updated continuously during real-time scan).
        self._scan_time_detect = 0.0
        self._scan_time_total = 0.0
        self._scan_fps = 0.0
        self._frame_count = 0

        # Tkinter state — initialised in run().
        self._root: tk.Tk | None = None
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self._preview_label: ttk.Label | None = None
        self._result_label: ttk.Label | None = None
        self._debug_text: tk.Text | None = None
        self._scan_btn: ttk.Button | None = None

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _ensure_sct(self) -> Any:
        if self._sct is None:
            import mss  # pylint: disable=import-outside-toplevel

            self._sct = mss.mss()
        return self._sct

    def _capture_frame(self) -> np.ndarray:
        """Grab a single frame from the configured monitor/region."""
        sct = self._ensure_sct()
        if self._region is not None:
            x, y, w, h = self._region
            grab_area = {"left": x, "top": y, "width": w, "height": h}
        else:
            grab_area = sct.monitors[self._monitor_index]

        screenshot = sct.grab(grab_area)
        frame = np.array(screenshot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------
    # Continuous CNN scanning (real-time loop)
    # ------------------------------------------------------------------

    def _ensure_cnn_resources(self) -> None:
        """Lazily create CNN embedder and FAISS index."""
        if self._embedder is None:
            # pylint: disable=import-outside-toplevel
            from card_reco.embedder import CardEmbedder

            self._embedder = CardEmbedder()
        if self._card_index is None:
            # pylint: disable=import-outside-toplevel
            from card_reco.faiss_index import CardIndex

            self._card_index = CardIndex()
        self._cnn_ready.set()

    def _preload_cnn(self) -> None:
        """Load CNN resources in a background thread."""
        self._ensure_cnn_resources()
        if self._root is not None:
            self._root.after(0, self._on_cnn_loaded)

    def _on_cnn_loaded(self) -> None:
        """Update UI after CNN resources finish loading (main thread)."""
        if self._debug_text is not None:
            self._debug_text.delete("1.0", tk.END)
            self._debug_text.insert(tk.END, "CNN model loaded. Ready to scan.\n")

    def _scan_loop(self) -> None:
        """Continuous detect-identify-track loop (background thread)."""
        self._cnn_ready.wait()
        assert self._embedder is not None
        assert self._card_index is not None

        last_seq = -1
        while self._scanning:
            # Grab the latest frame captured by the main-thread preview.
            with self._lock:
                frame = self._current_frame
                seq = self._frame_seq

            if frame is None or seq == last_seq:
                time.sleep(0.01)
                continue
            last_seq = seq

            t0 = time.perf_counter()
            detections = detect_cards(frame, max_detect_dim=1024, fast=True)
            t_detect = time.perf_counter() - t0

            pairs = identify_detections(
                detections,
                embedder=self._embedder,
                card_index=self._card_index,
                top_n=self._top_n,
            )
            t_total = time.perf_counter() - t0

            self._tracker.update(pairs)

            with self._lock:
                self._scan_time_detect = t_detect
                self._scan_time_total = t_total
                self._scan_fps = 1.0 / max(t_total, 0.001)
                self._frame_count += 1

            if self._root is not None:
                self._root.after(0, self._refresh_display)

    # ------------------------------------------------------------------
    # Scan button handling
    # ------------------------------------------------------------------

    def _on_scan(self) -> None:
        """Toggle continuous scanning on/off."""
        assert self._scan_btn is not None
        assert self._debug_text is not None

        if self._scanning:
            self._stop_scanning()
        else:
            self._start_scanning()

    def _start_scanning(self) -> None:
        assert self._scan_btn is not None
        assert self._debug_text is not None

        self._scanning = True
        self._frame_count = 0
        self._tracker.clear()
        self._scan_btn.configure(text="Stop")
        self._debug_text.delete("1.0", tk.END)
        if not self._cnn_ready.is_set():
            self._debug_text.insert(
                tk.END, "Waiting for CNN model to finish loading...\n"
            )
        else:
            self._debug_text.insert(tk.END, "Starting real-time scan...\n")

        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()

    def _stop_scanning(self) -> None:
        assert self._scan_btn is not None
        self._scanning = False
        self._scan_btn.configure(text="Scan")
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=2.0)
            self._scan_thread = None

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Periodically capture a frame and update the preview panel."""
        if self._root is None or self._preview_label is None:
            return
        try:
            frame = self._capture_frame()
            with self._lock:
                self._current_frame = frame
                self._frame_seq += 1

            vis = frame
            # When scanning, overlay tracked cards on the preview.
            if self._scanning:
                display = self._tracker.get_display_data()
                if display:
                    vis = self._draw_tracked(frame, display)

            small = _scale_image(vis, _PANEL_MAX_W, _PANEL_MAX_H)
            photo = _bgr_to_photoimage(small)
            self._preview_label.configure(image=photo)
            self._photo_refs["preview"] = photo
        except Exception:  # pylint: disable=broad-except
            pass  # capture errors are non-fatal

        self._root.after(_PREVIEW_INTERVAL_MS, self._refresh_preview)

    def _refresh_display(self) -> None:
        """Update the result panel and debug text from scan loop data."""
        if self._result_label is None or self._debug_text is None:
            return
        if self._scan_btn is None:
            return

        with self._lock:
            frame = self._current_frame
            t_detect = self._scan_time_detect
            t_total = self._scan_time_total
            fps = self._scan_fps
            frame_count = self._frame_count

        display = self._tracker.get_display_data()

        # --- Result image ---
        if frame is not None and display:
            vis = self._draw_tracked(frame, display)
            small = _scale_image(vis, _PANEL_MAX_W, _PANEL_MAX_H)
            photo = _bgr_to_photoimage(small)
            self._result_label.configure(image=photo)
            self._photo_refs["result"] = photo

        # --- Debug text ---
        self._debug_text.delete("1.0", tk.END)
        self._debug_text.insert(tk.END, f"Detection: {t_detect:.3f}s\n")
        self._debug_text.insert(tk.END, f"Total:     {t_total:.3f}s\n")
        self._debug_text.insert(
            tk.END, f"FPS:       {fps:.1f}  (frame #{frame_count})\n"
        )
        self._debug_text.insert(tk.END, f"Tracked:   {self._tracker.count} cards\n")
        self._debug_text.insert(tk.END, "-" * 40 + "\n")

        for i, (_corners, matches) in enumerate(display):
            self._debug_text.insert(tk.END, f"\nCard {i + 1}:\n")
            if not matches:
                self._debug_text.insert(tk.END, "  No match\n")
                continue
            for m in matches:
                c = m.card
                line = (
                    f"  #{m.rank} {c.name} ({c.set_name} {c.number})"
                    f" [sim={m.distance:.3f}]\n"
                )
                self._debug_text.insert(tk.END, line)

    @staticmethod
    def _draw_tracked(
        frame: np.ndarray,
        display: list[tuple[np.ndarray, list[MatchResult]]],
    ) -> np.ndarray:
        """Draw tracked card overlays on *frame*."""
        vis = frame.copy()
        for i, (corners, matches) in enumerate(display):
            color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
            pts = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, color, 2)

            if matches:
                label = matches[0].card.name
                sim = matches[0].distance
                text = f"{label} ({sim:.2f})"
            else:
                text = "No match"

            tl = corners[0].astype(int)
            cv2.putText(
                vis,
                text,
                (tl[0], max(tl[1] - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return vis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Launch the scanner GUI (blocks until the window is closed)."""
        root = tk.Tk()
        root.title("Card Scanner")
        self._root = root

        # ---- Top frame: preview (left) + result (right) ----
        top = ttk.Frame(root)
        top.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        preview_frame = ttk.LabelFrame(top, text="Live Preview")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self._preview_label = ttk.Label(preview_frame)
        self._preview_label.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(top, text="Last Scan Result")
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self._result_label = ttk.Label(result_frame)
        self._result_label.pack(fill=tk.BOTH, expand=True)

        # ---- Bottom frame: debug text + scan button ----
        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.BOTH, padx=5, pady=(0, 5))

        debug_frame = ttk.LabelFrame(bottom, text="Debug Info")
        debug_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        self._debug_text = tk.Text(debug_frame, height=12, width=60, wrap=tk.WORD)
        self._debug_text.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(bottom)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))

        self._scan_btn = ttk.Button(btn_frame, text="Scan", command=self._on_scan)
        self._scan_btn.pack(pady=10, ipadx=20, ipady=10)

        # Pre-load CNN resources in background so the GUI stays responsive.
        if self._backend == "cnn":
            self._debug_text.insert(tk.END, "Loading CNN model and card index...\n")
            threading.Thread(target=self._preload_cnn, daemon=True).start()

        # Start live preview loop.
        self._root.after(0, self._refresh_preview)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.mainloop()

    def close(self) -> None:
        """Release resources."""
        self._scanning = False
        if self._scan_thread is not None:
            self._scan_thread.join(timeout=2.0)
            self._scan_thread = None
        self._embedder = None
        self._card_index = None
        if self._sct is not None:
            self._sct.__exit__(None, None, None)
            self._sct = None

    def _on_close(self) -> None:
        """Clean up resources and close the window."""
        self.close()
        if self._root is not None:
            self._root.destroy()
            self._root = None
