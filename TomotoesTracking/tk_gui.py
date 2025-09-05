import os
import threading
import time
import base64
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from PIL import Image, ImageTk  # type: ignore
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False
from main import draw_boxes_no_id, draw_stats_panel 


@dataclass
class AppState:
    input_path: str = ""
    output_path: str = ""
    model_path: str = "best.pt"
    fps_out: Optional[float] = None
    running: bool = False
    stop_flag: bool = False


class YOLOTkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Tracking - Tkinter GUI")
        self.state = AppState()

        # UI layout
        self._build_ui()

        # Video / model resources
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.model = None
        self.names = None
        self.proc_thread: Optional[threading.Thread] = None

        # For keeping a reference to the PhotoImage
        self._tk_img = None

    def _build_ui(self):
        pad = dict(padx=8, pady=6)

        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True)

        # Row 1: Input
        row1 = ttk.Frame(frm)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text="Input video:").pack(side=tk.LEFT)
        self.in_var = tk.StringVar()
        in_entry = ttk.Entry(row1, textvariable=self.in_var, width=60)
        in_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(row1, text="Browse", command=self._pick_input).pack(side=tk.LEFT)

        # Row 2: Output
        row2 = ttk.Frame(frm)
        row2.pack(fill=tk.X, **pad)
        ttk.Label(row2, text="Output video:").pack(side=tk.LEFT)
        self.out_var = tk.StringVar(value="tracked.mp4")
        out_entry = ttk.Entry(row2, textvariable=self.out_var, width=60)
        out_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(row2, text="Save As", command=self._pick_output).pack(side=tk.LEFT)

        # Row 3: Model
        row3 = ttk.Frame(frm)
        row3.pack(fill=tk.X, **pad)
        ttk.Label(row3, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="best.pt")
        model_entry = ttk.Entry(row3, textvariable=self.model_var, width=40)
        model_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(row3, text="Browse", command=self._pick_model).pack(side=tk.LEFT)

        # Row 4: Controls
        row4 = ttk.Frame(frm)
        row4.pack(fill=tk.X, **pad)
        self.start_btn = ttk.Button(row4, text="Start", command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(row4, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        ttk.Label(row4, text="FPS out (opt):").pack(side=tk.LEFT, padx=(12, 4))
        self.fps_var = tk.StringVar()
        ttk.Entry(row4, textvariable=self.fps_var, width=8).pack(side=tk.LEFT)

        # Row 5: Status
        row5 = ttk.Frame(frm)
        row5.pack(fill=tk.X, **pad)
        self.status_var = tk.StringVar(value="Idle")
        self.counts_var = tk.StringVar(value="Counts :")
        ttk.Label(row5, textvariable=self.status_var).pack(side=tk.LEFT)

        # Row 6: Preview canvas
        canvas_frm = ttk.Frame(frm)
        canvas_frm.pack(fill=tk.BOTH, expand=True, **pad)
        # Default preview size
        self.preview_w = 960
        self.preview_h = 540
        self.canvas = tk.Label(canvas_frm, background="#000000")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # On close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI callbacks ----------
    def _pick_input(self):
        path = filedialog.askopenfilename(
            title="Choose input video",
            filetypes=[("Video", "*.mp4;*.mov;*.avi;*.mkv;*.MOV"), ("All files", "*.*")],
        )
        if path:
            self.in_var.set(path)

    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")],
            initialfile="tracked.mp4",
        )
        if path:
            self.out_var.set(path)

    def _pick_model(self):
        path = filedialog.askopenfilename(
            title="Choose YOLO model",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _start(self):
        if self.state.running:
            return
        inp = self.in_var.get().strip()
        outp = self.out_var.get().strip()
        modelp = self.model_var.get().strip() or "best.pt"
        fps_txt = (self.fps_var.get() or "").strip()
        fps_out = None
        if fps_txt:
            try:
                fps_out = float(fps_txt)
            except ValueError:
                messagebox.showerror("Invalid FPS", "FPS out must be a number")
                return

        if not inp or not os.path.exists(inp):
            messagebox.showerror("Input missing", "Please select a valid input video")
            return
        if not outp:
            messagebox.showerror("Output missing", "Please choose an output path")
            return

        self.state = AppState(inp, outp, modelp, fps_out, True, False)
        self._set_running_ui(True)
        self.status_var.set("Loading model...")

        self.proc_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.proc_thread.start()

    def _stop(self):
        self.state.stop_flag = True
        self.status_var.set("Stopping...")

    def _set_running_ui(self, running: bool):
        self.start_btn.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL if running else tk.DISABLED)

    def _on_close(self):
        if self.state.running:
            self._stop()
            # Give the thread a moment to exit cleanly
            self.root.after(300, self.root.destroy)
        else:
            self.root.destroy()

    # ---------- Processing pipeline ----------
    def _load_model(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            messagebox.showerror(
                "Ultralytics missing",
                "Ultralytics is not installed. Install with: pip install ultralytics\n"
                "Note: You also need PyTorch. See https://pytorch.org/get-started/locally/",
            )
            raise

        model = YOLO(self.state.model_path)
        names = model.model.names if hasattr(model, "model") else model.names
        return model, names

    def _run_pipeline(self):
        try:
            cap = cv2.VideoCapture(self.state.input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.state.input_path}")

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fps_out = self.state.fps_out if self.state.fps_out is not None else src_fps

            # Load model
            model, names = self._load_model()
            self.model, self.names = model, names

            # Prepare writer
            writer = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            last_ui = 0.0
            unique_ids_by_cls = {}

            while not self.state.stop_flag:
                ok, frame = cap.read()
                if not ok:
                    break

                res = model.track(frame, persist=True)
                r = res[0]
                boxes = r.boxes

                # Unique counts
                for b in boxes:
                    if b.id is not None:
                        tid = int(b.id[0])
                        cls = int(b.cls[0])
                        s = unique_ids_by_cls.setdefault(cls, set())
                        s.add(tid)

                draw_boxes_no_id(frame, boxes, names)
                draw_stats_panel(frame, unique_ids_by_cls, names)

                # Init writer lazily
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(self.state.output_path, fourcc, fps_out, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to init VideoWriter: {self.state.output_path}")

                writer.write(frame)

                # Throttle UI updates a bit
                now = time.time()
                if now - last_ui > 0.02:  # ~50 FPS UI max
                    last_ui = now
                    self._update_preview(frame)
                    self._update_counts(unique_ids_by_cls, names)

            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()

            # Final UI state
            self.root.after(0, lambda: self.status_var.set("Done"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.state.running = False
            self.root.after(0, lambda: self._set_running_ui(False))

    # ---------- UI update helpers ----------
    def _resize_keep_aspect(self, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        return max(1, int(w * scale)), max(1, int(h * scale))

    def _cv_to_photoimage(self, frame_bgr) -> tk.PhotoImage:
    
        h, w = frame_bgr.shape[:2]
        lbl_w = max(self.canvas.winfo_width(), 1)
        lbl_h = max(self.canvas.winfo_height(), 1)
        new_w, new_h = self._resize_keep_aspect(w, h, lbl_w, lbl_h)
        frame_resized = frame_bgr if (new_w == w and new_h == h) else cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=im)


    def _update_preview(self, frame_bgr):
        try:
            img = self._cv_to_photoimage(frame_bgr)
            # Keep a reference to avoid GC
            self._tk_img = img
            self.root.after(0, lambda: self.canvas.config(image=img))
            self.root.after(0, lambda: self.status_var.set("Running... (ESC in app not needed)"))
        except Exception as e:
            # Show once, but do not spam
            self.root.after(0, lambda: self.status_var.set(f"Preview error: {e}"))

    def _update_counts(self, unique_ids_by_cls, names):
        try:
            parts = ["Counts :"] + [f"{names[c]}: {len(s)}" for c, s in sorted(unique_ids_by_cls.items())]
            txt = "  |  ".join(parts)
            self.root.after(0, lambda: self.status_var.set(txt))
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = YOLOTkApp(root)
    root.geometry("1100x700")
    root.minsize(800, 500)
    root.mainloop()


if __name__ == "__main__":
    main()
