#!/usr/bin/env python3
"""
Silent Speech Recognition - Main Application
Vision-Based Lip Reading System

Usage:
    python main.py                          # Webcam, default settings
    python main.py --source webcam          # Explicit webcam
    python main.py --source video --input path/to/video.mp4
    python main.py --device cpu             # Force CPU
    python main.py --save-output            # Save predictions to file
    python main.py --beam-search            # Use beam search decoding
    python main.py --debug                  # Show debug info
    python main.py --no-display             # Headless mode (save video only)
"""

import argparse
import sys
import time
import logging
import signal
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face_detector import FaceDetector, FaceDetection
from src.lip_extractor import LipExtractor, LipRegion
from src.inference_engine import InferenceEngine, InferenceResult
from src.video_capture import VideoSource, VideoWriter
from src.display import OverlayRenderer, DisplayConfig, WindowManager


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Suppress noisy library logs
    for logger_name in ["mediapipe", "urllib3", "PIL"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Silent Speech Recognition - Vision-Based Lip Reading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Source options
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        choices=["webcam", "video"],
        help="Input source (default: webcam)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input video file path (required for --source video)"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )

    # Model options
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Compute device (default: auto-detect)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/lipnet_model.pth",
        help="Path to model weights"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=75,
        help="Frames per inference sequence (default: 75)"
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=15,
        help="Minimum frames before inference (default: 15)"
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=8,
        help="Run inference every N frames (default: 8)"
    )
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Use beam search decoding (slower but more accurate)"
    )

    # Output options
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save predicted text to file"
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default="output/predictions.txt",
        help="Output text file path"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save processed video to file"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="output/output.mp4",
        help="Output video file path"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode (no window display)"
    )

    # Display options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information"
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Show lip landmarks"
    )
    parser.add_argument(
        "--no-lip-preview",
        action="store_true",
        help="Hide lip region preview"
    )

    return parser.parse_args()


class SilentSpeechApp:
    """
    Main application class for Silent Speech Recognition.
    Orchestrates all components for real-time lip reading.
    """

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self._saved_text_count = 0

        # Setup output directory
        Path("output").mkdir(exist_ok=True)

        # Initialize components
        self.logger.info("Initializing components...")
        self._init_components()

    def _init_components(self):
        """Initialize all pipeline components."""
        args = self.args

        # 1. Face Detector
        self.logger.info("[1/6] Initializing face detector...")
        self.face_detector = FaceDetector(
            min_detection_confidence=0.5,
            model_selection=0
        )

        # 2. Lip Extractor
        self.logger.info("[2/6] Initializing lip extractor...")
        self.lip_extractor = LipExtractor(
            target_size=(112, 112),
            padding_factor=0.4,
            use_face_alignment=True,
            smooth_landmarks=True
        )

        # 3. Inference Engine
        self.logger.info("[3/6] Loading inference engine (this may take a moment)...")
        weights_path = Path(args.weights)
        weights_dir = weights_path.parent

        try:
            self.engine = InferenceEngine(
                weights_dir=weights_dir,
                device=args.device,
                sequence_length=args.sequence_length,
                min_frames=args.min_frames,
                inference_interval=args.inference_interval,
                use_beam_search=args.beam_search,
                beam_width=5
            )
        except FileNotFoundError:
            self.logger.error(
                "\n" + "="*60 + "\n"
                "Model weights not found!\n"
                "Please run setup first:\n"
                "  python setup_weights.py\n"
                "="*60
            )
            sys.exit(1)

        # 4. Video Source
        self.logger.info("[4/6] Opening video source...")
        self.video_source = VideoSource(
            source=args.source,
            device_id=args.camera_id,
            video_path=args.input,
            target_fps=30.0,
            width=640,
            height=480
        )

        if not self.video_source.open():
            self.logger.error("Failed to open video source")
            sys.exit(1)

        # 5. Display
        self.logger.info("[5/6] Setting up display...")
        display_config = DisplayConfig(
            show_lip_crop=not args.no_lip_preview,
            show_face_bbox=True,
            show_lip_landmarks=args.show_landmarks,
            show_debug_info=True,
            show_confidence_bar=True
        )
        self.renderer = OverlayRenderer(config=display_config)

        if not args.no_display:
            self.window_manager = WindowManager("Silent Speech Recognition")
            self.window_manager.create()
        else:
            self.window_manager = None

        # 6. Optional video writer
        self.video_writer = None
        if args.save_video:
            self.logger.info("[6/6] Setting up video writer...")
            w, h = self.video_source.get_resolution()
            self.video_writer = VideoWriter(
                output_path=args.output_video,
                fps=30.0,
                width=w,
                height=h
            )
            if not self.video_writer.open():
                self.logger.warning("Could not open video writer")
                self.video_writer = None
        else:
            self.logger.info("[6/6] Video writer disabled")

        self.logger.info("All components initialized successfully!")

    def run(self):
        """Run the main processing loop."""
        self.logger.info("Starting main loop (press Q to quit)...")
        self._running = True

        # Statistics
        loop_start = time.perf_counter()
        total_frames = 0
        inference_results = []
        last_save_time = time.perf_counter()
        save_interval = 5.0  # Save text every 5 seconds

        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            while self._running:
                # Read frame
                ret, frame = self.video_source.read_direct()

                if not ret or frame is None:
                    self.logger.warning("Empty frame received")
                    time.sleep(0.01)
                    continue

                total_frames += 1

                # --- PIPELINE ---

                # Step 1: Face Detection
                face_detection: Optional[FaceDetection] = self.face_detector.detect(frame)

                # Step 2: Lip Region Extraction
                lip_region: Optional[LipRegion] = self.lip_extractor.extract(frame)

                # Step 3-6: Preprocessing + Inference + Decoding
                result: Optional[InferenceResult] = self.engine.process_lip_region(lip_region)

                # --- OUTPUT ---

                # Render overlays
                display_frame = self.renderer.render(
                    frame=frame,
                    face_detection=face_detection,
                    lip_region=lip_region,
                    inference_result=result,
                    show_controls=True
                )

                # Display
                if self.window_manager is not None:
                    key = self.window_manager.show(display_frame)

                    # In SilentSpeechApp.run(), inside the while loop
    # AFTER:  key = self.window_manager.show(display_frame)
    # ADD this line:

                    # Print prediction to terminal when in no-GUI mode
                    if result is not None and self.window_manager is not None:
                        self.window_manager.print_prediction(
                            result.stable_text or result.text,
                            result.confidence
                        )

                    # Handle keyboard input
                    should_quit = self._handle_key(key, result)
                    if should_quit:
                        break

                    # Check if window was closed
                    if not self.window_manager.is_open():
                        self.logger.info("Window closed by user")
                        break

                # Save video
                if self.video_writer is not None:
                    self.video_writer.write(display_frame)

                # Auto-save text output
                if (self.args.save_output and
                    result is not None and
                    result.stable_text and
                    time.perf_counter() - last_save_time > save_interval):

                    self._save_prediction(result)
                    last_save_time = time.perf_counter()

                # Print to console periodically
                if total_frames % 90 == 0:
                    self._print_status(result, total_frames, loop_start)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self._cleanup(total_frames, loop_start)

    def _handle_key(self, key: int, result: Optional[InferenceResult]) -> bool:
        """
        Handle keyboard input.

        Returns:
            True if should quit
        """
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            self.logger.info("Quit requested")
            return True

        elif key == ord('r') or key == ord('R'):  # Reset
            self.engine.reset()
            self.renderer.aggregator.reset() if hasattr(self.renderer, 'aggregator') else None
            self.logger.info("Pipeline reset")

        elif key == ord('s') or key == ord('S'):  # Save
            if result is not None:
                self._save_prediction(result)
                self.logger.info(f"Saved: '{result.stable_text}'")

        elif key == ord('d') or key == ord('D'):  # Toggle debug
            self.renderer.toggle_debug()

        elif key == ord('l') or key == ord('L'):  # Toggle landmarks
            self.renderer.toggle_landmarks()

        elif key == ord('p') or key == ord('P'):  # Print stats
            stats = self.engine.get_performance_stats()
            self.logger.info(f"Performance: {stats}")

        return False

    def _handle_signal(self, signum, frame):
        """Handle SIGINT signal."""
        self.logger.info("Signal received, stopping...")
        self._running = False

    def _save_prediction(self, result: InferenceResult):
        """Save prediction to text file."""
        text_path = Path(self.args.output_text)
        text_path.parent.mkdir(parents=True, exist_ok=True)

        with open(text_path, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            text = result.stable_text or result.text
            confidence = result.confidence
            f.write(f"[{timestamp}] {text} (confidence: {confidence:.2%})\n")

        self._saved_text_count += 1

    def _print_status(
        self,
        result: Optional[InferenceResult],
        total_frames: int,
        start_time: float
    ):
        """Print status to console."""
        elapsed = time.perf_counter() - start_time
        fps = total_frames / max(elapsed, 1)
        stats = self.engine.get_performance_stats()

        print(
            f"\r[Frame {total_frames:5d}] "
            f"FPS: {fps:5.1f} | "
            f"Infer: {stats.get('avg_ms', 0):5.1f}ms | "
            f"Text: '{result.stable_text if result else '':<30}' | "
            f"Conf: {result.confidence:.0%}" if result else "",
            end="",
            flush=True
        )

    def _cleanup(self, total_frames: int, start_time: float):
        """Clean up resources and print final statistics."""
        self.logger.info("\nCleaning up...")

        # Stop video source
        self.video_source.stop()

        # Close window
        if self.window_manager is not None:
            self.window_manager.destroy()
        cv2.destroyAllWindows()

        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.logger.info(f"Video saved to: {self.args.output_video}")

        # Final statistics
        elapsed = time.perf_counter() - start_time
        avg_fps = total_frames / max(elapsed, 1)
        stats = self.engine.get_performance_stats()

        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed : {total_frames}")
        print(f"Total time             : {elapsed:.1f}s")
        print(f"Average FPS            : {avg_fps:.1f}")
        print(f"Total inferences       : {stats.get('inference_count', 0)}")
        print(f"Avg inference time     : {stats.get('avg_ms', 0):.1f}ms")
        print(f"Text predictions saved : {self._saved_text_count}")

        # Print text history
        history = self.renderer.get_history()
        if history:
            print("\nPredicted text history:")
            for i, text in enumerate(history, 1):
                print(f"  {i}. {text}")

        print("="*60)

        self.logger.info("Cleanup complete")


def check_requirements():
    """Check that all required packages are available."""
    required = {
        "cv2": "opencv-python",
        "torch": "torch",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "torchvision": "torchvision"
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print("  uv pip install -e .")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger("main")

    # Banner
    print("="*60)
    print("  Silent Speech Recognition - Vision-Based Lip Reading")
    print("  Version 1.0.0")
    print("="*60)
    print(f"  Source  : {args.source}")
    print(f"  Device  : {args.device or 'auto'}")
    print(f"  Weights : {args.weights}")
    print(f"  Decoder : {'beam search' if args.beam_search else 'greedy'}")
    print("="*60)

    # Check requirements
    check_requirements()

    # Validate video source arguments
    if args.source == "video" and args.input is None:
        logger.error("--input is required when --source is 'video'")
        sys.exit(1)

    if args.source == "video" and not Path(args.input).exists():
        logger.error(f"Video file not found: {args.input}")
        sys.exit(1)

    # Check weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(
            f"\nModel weights not found at: {weights_path}\n"
            f"Please run setup first:\n"
            f"  python setup_weights.py"
        )
        sys.exit(1)

    # Create and run application
    try:
        app = SilentSpeechApp(args)
        app.run()
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()