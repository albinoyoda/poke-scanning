"""Live scanner GUI — capture screen, identify cards on demand."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from card_reco.detector import detect_cards
from card_reco.matcher import CardMatcher
from card_reco.pipeline import identify_cards_from_array

if TYPE_CHECKING:
    from card_reco.models import DetectedCard, MatchResult

# How often the live preview refreshes (milliseconds).
_PREVIEW_INTERVAL_MS = 200

# Maximum dimensions for preview and result panels.
_PANEL_MAX_W = 640
_PANEL_MAX_H = 480

# Shared colour palette for detection overlays (avoids pylint duplicate-code
# with debug.py and detector/).
_OVERLAY_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


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
            text = f"{label} ({dist:.1f})"
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


class Scanner:
    """Tkinter GUI for on-demand screen-capture card scanning."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        monitor: int = 1,
        region: tuple[int, int, int, int] | None = None,
        top_n: int = 5,
        threshold: float = 40.0,
    ) -> None:
        self._db_path = db_path
        self._monitor_index = monitor
        self._region = region
        self._top_n = top_n
        self._threshold = threshold

        # Matcher — created lazily on first scan.
        self._matcher: CardMatcher | None = None

        # Screen-capture context (lazily created).
        self._sct: Any = None

        # Latest captured frame (BGR numpy array).
        self._current_frame: np.ndarray | None = None

        # Results from the most recent scan.
        self._last_detections: list[DetectedCard] = []
        self._last_results: list[list[MatchResult]] = []
        self._scan_busy = False
        self._lock = threading.Lock()

        # Tkinter state — initialised in run().
        self._root: tk.Tk | None = None
        # Keep references to PhotoImage objects to prevent GC.
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        # Scan timing (set by _do_scan).
        self._scan_time_detect = 0.0
        self._scan_time_total = 0.0
        # Widget references — set in run().
        self._preview_label: ttk.Label | None = None
        self._result_label: ttk.Label | None = None
        self._debug_text: tk.Text | None = None
        self._scan_btn: ttk.Button | None = None

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _ensure_sct(self) -> Any:
        if self._sct is None:
            # Lazy import so the module can be imported without mss installed
            # (e.g. during type-checking or testing).
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
        # mss returns BGRA; drop the alpha channel.
        frame = np.array(screenshot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------
    # Scanning (runs in background thread)
    # ------------------------------------------------------------------

    def _do_scan(self, frame: np.ndarray) -> None:
        """Run the full pipeline on *frame* in a background thread."""
        try:
            if self._matcher is None:
                self._matcher = CardMatcher(self._db_path)

            t0 = time.perf_counter()
            detections = detect_cards(frame)
            t_detect = time.perf_counter() - t0

            results = identify_cards_from_array(
                frame,
                db_path=self._db_path,
                top_n=self._top_n,
                threshold=self._threshold,
                matcher=self._matcher,
            )
            t_total = time.perf_counter() - t0

            with self._lock:
                self._last_detections = detections
                self._last_results = results
                self._scan_time_detect = t_detect
                self._scan_time_total = t_total
        finally:
            with self._lock:
                self._scan_busy = False

            # Schedule a UI refresh on the main thread.
            if self._root is not None:
                self._root.after(0, self._refresh_results)

    def _on_scan(self) -> None:
        """Handle the Scan button press."""
        assert self._scan_btn is not None
        assert self._debug_text is not None
        with self._lock:
            if self._scan_busy:
                return
            self._scan_busy = True
            frame = self._current_frame

        if frame is None:
            with self._lock:
                self._scan_busy = False
            return

        self._scan_btn.configure(state="disabled")
        self._debug_text.delete("1.0", tk.END)
        self._debug_text.insert(tk.END, "Scanning…\n")
        thread = threading.Thread(
            target=self._do_scan,
            args=(frame.copy(),),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Periodically capture a frame and update the preview panel."""
        if self._root is None or self._preview_label is None:
            return
        try:
            frame = self._capture_frame()
            self._current_frame = frame
            small = _scale_image(frame, _PANEL_MAX_W, _PANEL_MAX_H)
            photo = _bgr_to_photoimage(small)
            self._preview_label.configure(image=photo)
            self._photo_refs["preview"] = photo
        except Exception:  # pylint: disable=broad-except
            pass  # capture errors are non-fatal

        self._root.after(_PREVIEW_INTERVAL_MS, self._refresh_preview)

    def _refresh_results(self) -> None:
        """Update the result panel and debug text after a scan completes."""
        assert self._scan_btn is not None
        assert self._result_label is not None
        assert self._debug_text is not None
        with self._lock:
            detections = list(self._last_detections)
            results = list(self._last_results)
            t_detect = self._scan_time_detect
            t_total = self._scan_time_total

        self._scan_btn.configure(state="normal")

        # --- Result image: original frame with overlaid detections ---
        if self._current_frame is not None and detections:
            vis = _draw_detections(self._current_frame, detections, results)
            small = _scale_image(vis, _PANEL_MAX_W, _PANEL_MAX_H)
            photo = _bgr_to_photoimage(small)
            self._result_label.configure(image=photo)
            self._photo_refs["result"] = photo

        # --- Debug text ---
        self._debug_text.delete("1.0", tk.END)
        self._debug_text.insert(tk.END, f"Detection: {t_detect:.3f}s\n")
        self._debug_text.insert(tk.END, f"Total:     {t_total:.3f}s\n")
        self._debug_text.insert(tk.END, f"Cards detected: {len(detections)}\n")
        self._debug_text.insert(tk.END, "-" * 40 + "\n")

        for i, matches in enumerate(results):
            self._debug_text.insert(tk.END, f"\nCard {i + 1}:\n")
            if not matches:
                self._debug_text.insert(tk.END, "  No match\n")
                continue
            for m in matches:
                c = m.card
                line = (
                    f"  #{m.rank} {c.name} ({c.set_name} {c.number}) "
                    f"[dist={m.distance:.1f}]\n"
                )
                self._debug_text.insert(tk.END, line)
                if m.distances:
                    parts = " ".join(f"{k}={v}" for k, v in m.distances.items())
                    self._debug_text.insert(tk.END, f"       {parts}\n")

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

        # Preview panel
        preview_frame = ttk.LabelFrame(top, text="Live Preview")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        self._preview_label = ttk.Label(preview_frame)
        self._preview_label.pack(fill=tk.BOTH, expand=True)

        # Result panel
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

        # Start live preview loop.
        self._root.after(0, self._refresh_preview)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.mainloop()

    def close(self) -> None:
        """Release resources (matcher, screen-capture context)."""
        if self._matcher is not None:
            self._matcher.close()
            self._matcher = None
        if self._sct is not None:
            self._sct.__exit__(None, None, None)
            self._sct = None

    def _on_close(self) -> None:
        """Clean up resources and close the window."""
        self.close()
        if self._root is not None:
            self._root.destroy()
            self._root = None
