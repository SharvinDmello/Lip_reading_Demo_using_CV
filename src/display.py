"""
Display Module — Wayland/Hyprland Compatible
Handles OpenCV window creation with:
1. Automatic XWayland detection and DISPLAY setup
2. Graceful fallback to frame-saving if no GUI available
3. Terminal-based text output as last resort
"""

import cv2
import numpy as np
import os
import sys
import time
import subprocess
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ── Environment setup (must happen before any cv2.imshow call) ──────────────

def _find_xwayland_display() -> Optional[str]:
    """Detect the active XWayland DISPLAY variable."""
    # Already set
    if os.environ.get("DISPLAY"):
        return os.environ["DISPLAY"]

    # Try common display numbers
    for n in range(5):
        lock = f"/tmp/.X{n}-lock"
        sock = f"/tmp/.X11-unix/X{n}"
        if os.path.exists(lock) or os.path.exists(sock):
            logger.info(f"Found XWayland display :{n}")
            return f":{n}"

    # Ask Hyprland via hyprctl
    try:
        result = subprocess.run(
            ["hyprctl", "monitors", "-j"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            # XWayland is running if hyprland is active; default to :0
            if os.path.exists("/tmp/.X11-unix/X0"):
                return ":0"
            if os.path.exists("/tmp/.X11-unix/X1"):
                return ":1"
    except Exception:
        pass

    return None


def setup_display_env() -> bool:
    """
    Configure environment for OpenCV GUI under Wayland/Hyprland.
    Returns True if display environment looks usable.
    """
    # Force X11 backend for Qt (used by some OpenCV builds)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("GDK_BACKEND", "x11")

    display = _find_xwayland_display()
    if display:
        os.environ["DISPLAY"] = display
        logger.info(f"Set DISPLAY={display}")
        return True

    logger.warning(
        "No XWayland DISPLAY found. "
        "Ensure xorg-xwayland is installed and XWayland is enabled in hyprland.conf"
    )
    return False


def probe_gui() -> bool:
    """
    Safely probe whether OpenCV GUI works in this environment.
    Returns True if a window can be created.
    """
    setup_display_env()
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False
    except Exception:
        return False


# ── Run environment setup at import time ────────────────────────────────────
_GUI_AVAILABLE = probe_gui()
if not _GUI_AVAILABLE:
    logger.warning(
        "OpenCV GUI not available. "
        "Falling back to frame-save + terminal output mode."
    )


# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DisplayConfig:
    show_lip_crop:       bool  = True
    show_face_bbox:      bool  = True
    show_lip_landmarks:  bool  = False
    show_debug_info:     bool  = True
    show_confidence_bar: bool  = True
    text_font:           int   = cv2.FONT_HERSHEY_SIMPLEX
    text_scale:          float = 0.8
    text_thickness:      int   = 2
    text_color:          Tuple[int,int,int] = (255, 255, 255)
    text_bg_color:       Tuple[int,int,int] = (0, 0, 0)
    text_bg_alpha:       float = 0.7
    face_bbox_color:     Tuple[int,int,int] = (0, 255, 0)
    lip_bbox_color:      Tuple[int,int,int] = (0, 0, 255)
    landmark_color:      Tuple[int,int,int] = (255, 0, 0)
    status_ok_color:     Tuple[int,int,int] = (0, 255, 0)
    status_warn_color:   Tuple[int,int,int] = (0, 165, 255)
    status_error_color:  Tuple[int,int,int] = (0, 0, 255)
    lip_preview_size:    Tuple[int,int] = (150, 150)


# ════════════════════════════════════════════════════════════════════════════

class OverlayRenderer:
    """
    Renders all visual overlays onto video frames.
    Pure NumPy/OpenCV drawing — works with or without a display.
    """

    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()
        self._text_history: List[str] = []
        self._max_history   = 5
        self._start_time    = time.perf_counter()
        self._frame_times:  List[float] = []
        self._prev_time     = 0.0

    # ── main render ─────────────────────────────────────────────────────────
    def render(
        self,
        frame:            np.ndarray,
        face_detection,
        lip_region,
        inference_result,
        show_controls:    bool = True,
    ) -> np.ndarray:
        display = frame.copy()
        h, w    = display.shape[:2]

        if self.config.show_face_bbox and face_detection is not None:
            self._draw_face_bbox(display, face_detection)

        if lip_region is not None and lip_region.face_detected:
            if self.config.show_face_bbox:
                self._draw_lip_bbox(display, lip_region)
            if self.config.show_lip_landmarks:
                self._draw_lip_landmarks(display, lip_region)

        self._draw_speech_text(display, inference_result, h, w)

        if self.config.show_debug_info:
            self._draw_status_panel(
                display, face_detection, lip_region, inference_result, h, w
            )

        if self.config.show_confidence_bar and inference_result is not None:
            self._draw_confidence_bar(display, inference_result, h, w)

        if (self.config.show_lip_crop
                and lip_region is not None
                and lip_region.face_detected):
            self._draw_lip_preview(display, lip_region, h, w)

        if inference_result is not None:
            self._draw_buffer_indicator(display, inference_result, h, w)

        if show_controls:
            self._draw_controls(display, h, w)

        self._update_fps()
        return display

    # ── drawing helpers ─────────────────────────────────────────────────────
    def _draw_face_bbox(self, frame, detection):
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x,y), (x+w, y+h), self.config.face_bbox_color, 2)
        self._draw_label(
            frame, f"Face {detection.confidence:.0%}",
            (x, y-8), self.config.face_bbox_color
        )

    def _draw_lip_bbox(self, frame, lip_region):
        x, y, w, h = lip_region.bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), self.config.lip_bbox_color, 2)
        self._draw_label(frame, "Lips", (x, y-8), self.config.lip_bbox_color)

    def _draw_lip_landmarks(self, frame, lip_region):
        for pt in lip_region.landmarks:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x,y), 2, self.config.landmark_color, -1)

    def _draw_speech_text(self, frame, result, h, w):
        panel_h = 80
        panel_y = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, self.config.text_bg_alpha,
                        frame, 1-self.config.text_bg_alpha, 0, frame)

        cv2.putText(frame, "PREDICTED SPEECH:",
                    (10, panel_y+20),
                    self.config.text_font, 0.45, (150,150,150), 1)

        if result is not None:
            txt = result.stable_text or result.text
            if not txt:
                txt = "..." if result.face_detected else "[No face detected]"
            col = (0,255,150) if result.confidence > 0.5 \
                else (0,220,255) if result.confidence > 0.2 \
                else (150,150,150)
        else:
            txt = "Initializing..."
            col = (150,150,150)

        ty = panel_y + 55
        cv2.putText(frame, txt, (11, ty+1),
                    self.config.text_font, self.config.text_scale,
                    (0,0,0), self.config.text_thickness+1)
        cv2.putText(frame, txt, (10, ty),
                    self.config.text_font, self.config.text_scale,
                    col, self.config.text_thickness)

    def _draw_status_panel(self, frame, face_det, lip_reg, result, h, w):
        px, py   = w-220, 10
        line_h   = 22
        padding  = 8
        panel_h  = 8*line_h + 2*padding
        overlay  = frame.copy()
        cv2.rectangle(overlay, (px-padding, py), (w-5, py+panel_h), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = py + padding + line_h

        def line(label, value, color):
            nonlocal y
            cv2.putText(frame, f"{label}:", (px, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
            cv2.putText(frame, value, (px+90, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += line_h

        fps   = self._get_fps()
        fcol  = (0,255,0) if fps>=20 else (0,165,255) if fps>=10 else (0,0,255)
        line("FPS", f"{fps:.1f}", fcol)

        face_ok = face_det is not None and face_det.confidence > 0.5
        line("Face",
             f"{'YES' if face_ok else 'NO'} {face_det.confidence:.0%}"
             if face_det else "NO",
             self.config.status_ok_color if face_ok
             else self.config.status_error_color)

        lip_ok = lip_reg is not None and lip_reg.face_detected
        line("Lips", "YES" if lip_ok else "NO",
             self.config.status_ok_color if lip_ok
             else self.config.status_error_color)

        if result is not None:
            fill  = result.buffer_fill * 100
            fcol2 = (0,255,0) if fill >= 80 else (0,165,255)
            line("Buffer", f"{fill:.0f}%", fcol2)
            conf  = result.confidence
            ccol  = (0,255,0) if conf>0.5 else (0,165,255) if conf>0.2 else (0,0,255)
            line("Conf",   f"{conf:.1%}", ccol)
            line("Infer",  f"{result.processing_time_ms:.0f}ms", (200,200,200))
            line("Frames", str(result.frame_count), (200,200,200))

        elapsed = int(time.perf_counter() - self._start_time)
        line("Time", f"{elapsed//60:02d}:{elapsed%60:02d}", (200,200,200))

    def _draw_confidence_bar(self, frame, result, h, w):
        bx, by, bw, bh = 10, h-100, 150, 8
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (50,50,50), -1)
        fw = int(bw * result.confidence)
        if fw > 0:
            c = result.confidence
            r = int(255*(1-2*c)) if c<0.5 else 0
            g = int(255*2*c)     if c<0.5 else 255
            cv2.rectangle(frame, (bx,by), (bx+fw,by+bh), (0,g,r), -1)
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (200,200,200), 1)
        cv2.putText(frame, f"Conf: {result.confidence:.0%}",
                    (bx+bw+5, by+bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    def _draw_lip_preview(self, frame, lip_region, h, w):
        ps = self.config.lip_preview_size
        if lip_region.crop is None or lip_region.crop.size == 0:
            return
        preview = cv2.resize(lip_region.crop, ps)
        px, py  = 10, 10
        border  = 2
        ph      = ps[1] + 2*border + 20
        pw      = ps[0] + 2*border
        cv2.rectangle(frame, (px-border, py-border),
                      (px+pw, py+ph), (30,30,30), -1)
        frame[py:py+ps[1], px:px+ps[0]] = preview
        cv2.putText(frame, "Lip ROI", (px, py+ps[1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        cv2.rectangle(frame, (px-border, py-border),
                      (px+ps[0]+border, py+ps[1]+border),
                      (0,150,255), border)

    def _draw_buffer_indicator(self, frame, result, h, w):
        bx, by, bw, bh = 5, h-190, 8, 100
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (50,50,50), -1)
        fh = int(bh * result.buffer_fill)
        if fh > 0:
            col = (0,255,0) if result.buffer_fill>=0.8 else (0,165,255)
            cv2.rectangle(frame, (bx, by+bh-fh), (bx+bw, by+bh), col, -1)
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (200,200,200), 1)
        cv2.putText(frame, "BUF", (bx-1, by-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180,180,180), 1)

    def _draw_controls(self, frame, h, w):
        controls = ["Q:Quit","R:Reset","S:Save","D:Debug","L:Lmarks"]
        y = 180
        for c in controls:
            cv2.putText(frame, c, (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
            y += 18

    def _draw_label(self, frame, text, pos, color,
                    font_scale=0.4, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw,th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = max(0,pos[0]), max(th+5, pos[1])
        cv2.rectangle(frame, (x-1,y-th-bl), (x+tw+1,y+bl), (0,0,0), -1)
        cv2.putText(frame, text, (x,y), font, font_scale, color, thickness)

    def _update_fps(self):
        now = time.perf_counter()
        if self._prev_time > 0:
            fps = 1.0 / max(now - self._prev_time, 1e-6)
            self._frame_times.append(fps)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
        self._prev_time = now

    def _get_fps(self) -> float:
        return float(np.mean(self._frame_times)) if self._frame_times else 0.0

    def add_to_history(self, text: str):
        if text and (not self._text_history or self._text_history[-1] != text):
            self._text_history.append(text)
            if len(self._text_history) > self._max_history:
                self._text_history.pop(0)

    def get_history(self) -> List[str]:
        return self._text_history.copy()

    def toggle_debug(self):
        self.config.show_debug_info = not self.config.show_debug_info

    def toggle_landmarks(self):
        self.config.show_lip_landmarks = not self.config.show_lip_landmarks


# ════════════════════════════════════════════════════════════════════════════

class WindowManager:
    """
    OpenCV window manager with full Wayland/Hyprland/XWayland support.

    Priority:
      1. XWayland (DISPLAY=:0)  — best experience
      2. Frame-save mode        — saves annotated frames to output/frames/
      3. Terminal text mode     — prints predictions to stdout
    """

    def __init__(self, window_name: str = "Silent Speech Recognition"):
        self.window_name  = window_name
        self._created     = False
        self._gui_ok      = _GUI_AVAILABLE
        self._frame_count = 0
        self._last_print  = 0.0
        self._save_dir    = Path("output") / "frames"
        self._last_text   = ""

    # ── public API ──────────────────────────────────────────────────────────
    def create(self, resizable: bool = True):
        if not self._gui_ok:
            self._print_no_gui_banner()
            self._save_dir.mkdir(parents=True, exist_ok=True)
            return

        try:
            flags = cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE
            cv2.namedWindow(self.window_name, flags)
            cv2.resizeWindow(self.window_name, 900, 600)
            self._created = True
            logger.info(f"Window created: {self.window_name}")
        except cv2.error as exc:
            logger.warning(f"Window creation failed: {exc}")
            self._gui_ok = False
            self._print_no_gui_banner()
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def show(self, frame: np.ndarray) -> int:
        """Display frame. Returns keycode or -1."""
        self._frame_count += 1

        if self._gui_ok and self._created:
            try:
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                return key
            except cv2.error:
                self._gui_ok = False

        # ── Fallback: save every 15th frame ─────────────────────────────
        if self._frame_count % 15 == 0:
            path = self._save_dir / f"frame_{self._frame_count:07d}.jpg"
            cv2.imwrite(str(path), frame)

        return -1

    def print_prediction(self, text: str, confidence: float):
        """Print prediction to terminal (used in no-GUI mode)."""
        if not self._gui_ok and text and text != self._last_text:
            now = time.perf_counter()
            if now - self._last_print > 1.0:
                bar    = "█" * int(confidence * 20)
                empty  = "░" * (20 - int(confidence * 20))
                print(f"\r  [{bar}{empty}] {confidence:.0%}  »  {text:<50}",
                      end="", flush=True)
                self._last_print = now
                self._last_text  = text

    def destroy(self):
        if self._created and self._gui_ok:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._created = False

    def is_open(self) -> bool:
        if not self._gui_ok:
            return True     # headless always "open"
        try:
            return cv2.getWindowProperty(
                self.window_name, cv2.WND_PROP_VISIBLE
            ) >= 1
        except Exception:
            return False

    @property
    def gui_available(self) -> bool:
        return self._gui_ok

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _print_no_gui_banner():
        print("\n" + "═"*60)
        print("  NO GUI MODE  (Wayland / no X11 display)")
        print("  Annotated frames → output/frames/")
        print("  Predictions printed to terminal in real-time")
        print()
        print("  To enable the GUI window:")
        print("  1. Ensure XWayland is running:")
        print("       hyprctl monitors   # should list outputs")
        print("  2. Set DISPLAY in fish:")
        print("       set -x DISPLAY :0")
        print("       python main.py --source webcam")
        print("  3. Or add to ~/.config/fish/config.fish:")
        print("       set -x DISPLAY :0")
        print("       set -x QT_QPA_PLATFORM xcb")
        print("═"*60 + "\n")

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        cv2.destroyAllWindows()