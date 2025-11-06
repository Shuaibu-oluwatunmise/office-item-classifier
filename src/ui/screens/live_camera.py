import tkinter as tk
from threading import Thread
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from components.widgets import header, Card, Button, label, heading, spacer
from components.styles import COLORS, SPACE
from ultralytics import YOLO

class LiveCameraScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, on_done, on_cancel):
        """
        Live camera inference screen - opens OpenCV window for speed
        on_done() -> return to input screen
        """
        super().__init__(parent, bg=COLORS["bg_darkest"])
        self.mode = mode
        self.model_path = model_path
        self.on_done = on_done
        self.on_cancel = on_cancel
        
        self.is_running = False
        self.cap = None
        self.model = None
        self.font = None

        header(self, f"Live {mode.title()}").pack(fill="x")

        # Main content
        content = tk.Frame(self, bg=COLORS["bg_darkest"])
        content.pack(fill="both", expand=True, padx=SPACE["xl"], pady=SPACE["xl"])

        # Control panel at top
        control_card = Card(content, shadow="md", glow=True, padding=SPACE["md"])
        control_card.pack(fill="x", pady=(0, SPACE["lg"]))

        control_row = tk.Frame(control_card.outer, bg=control_card.outer.cget("bg"))
        control_row.pack(fill="x")

        # Status indicator
        status_frame = tk.Frame(control_row, bg=control_row.cget("bg"))
        status_frame.pack(side="left")
        
        self.status_dot = tk.Label(
            status_frame,
            text="●",
            fg=COLORS["danger"],
            bg=status_frame.cget("bg"),
            font=("Arial", 16)
        )
        self.status_dot.pack(side="left")
        
        self.status_label = label(
            status_frame,
            "Initializing...",
            variant="secondary",
            weight="bold",
            size=11
        )
        self.status_label.pack(side="left", padx=(SPACE["sm"], 0))

        # Control buttons
        btn_frame = tk.Frame(control_row, bg=control_row.cget("bg"))
        btn_frame.pack(side="right")

        self.btn_stop = Button(
            btn_frame,
            text="⏹ Stop Stream",
            command=self._stop_stream,
            variant="danger",
            size="md"
        )
        self.btn_stop.pack(side="right", padx=(SPACE["sm"], 0))

        Button(
            btn_frame,
            text="← Back",
            command=self._cancel,
            variant="dark",
            size="md"
        ).pack(side="right")

        # Info card explaining OpenCV window
        info_card = Card(content, shadow="md", padding=SPACE["lg"])
        info_card.pack(fill="both", expand=True)

        # Centered info
        center = tk.Frame(info_card.outer, bg=info_card.outer.cget("bg"))
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Camera icon
        icon = label(center, "📹", variant="neon", size=64)
        icon.pack()

        spacer(center, SPACE["xl"]).pack()

        heading(center, "Live Camera Window", level=1).pack()
        spacer(center, SPACE["sm"]).pack()

        label(
            center,
            "Camera will open in a separate window for better performance",
            variant="secondary",
            size=12
        ).pack()

        spacer(center, SPACE["md"]).pack()

        self.info_label = label(
            center,
            "Press 'Q' in camera window to stop",
            variant="neon",
            weight="bold",
            size=11
        )
        self.info_label.pack()

        # Start the camera in background
        Thread(target=self._init_camera, daemon=True).start()

    def _init_camera(self):
        """Initialize camera and model in background"""
        try:
            self.after(0, lambda: self.status_label.config(text="Loading model..."))
            
            # Load model
            self.model = YOLO(str(self.model_path))
            
            # Load font for classification
            if self.mode == "classification":
                try:
                    self.font = ImageFont.truetype("arial.ttf", 36)
                except:
                    self.font = None
            
            self.after(0, lambda: self.status_label.config(text="Opening camera..."))
            
            # My Laptop uses 1 as default
            self.cap = cv2.VideoCapture(1)

            # Try different camera indices
            # for idx in [0, 1, 2]:
            #     self.cap = cv2.VideoCapture(idx)
            #     if self.cap.isOpened():
            #         break
            
            if not self.cap or not self.cap.isOpened():
                self.after(0, lambda: self._camera_error("Could not open camera"))
                return
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            self.after(0, lambda: self.status_dot.config(fg=COLORS["primary"]))
            self.after(0, lambda: self.status_label.config(text="● LIVE", fg=COLORS["primary"]))
            
            # Start video loop in separate thread (OpenCV window)
            self._video_loop_opencv()
            
        except Exception as e:
            self.after(0, lambda: self._camera_error(f"Error: {str(e)}"))

    def _video_loop_opencv(self):
        """Main video processing loop using OpenCV window (MUCH FASTER)"""
        window_name = f"Live {self.mode.title()} - Press Q to stop"
        
        # Create resizable window and set it LARGE
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)  
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model.predict(frame, verbose=False)
            r = results[0]
            
            # Process based on mode
            if self.mode == "classification":
                # Top-1 classification only
                if r.probs.top1conf > 0.3:
                    class_name = self.model.names[r.probs.top1]
                    conf = r.probs.top1conf.item() * 100
                    label_text = f"{class_name} - {conf:.2f}%"
                    
                    # Draw text overlay with BIGGER font
                    cv2.rectangle(frame, (10, 10), (700, 100), (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (30, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 135), 4, cv2.LINE_AA)
            
            else:  # detection
                # Draw bounding boxes
                frame = r.plot()
            
            # Show frame in OpenCV window
            cv2.imshow(window_name, frame)
            
            # Check for 'q' key or window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()
        
        self.is_running = False
        self.after(0, self._stop_stream)

    def _camera_error(self, message):
        """Handle camera errors"""
        self.status_dot.config(fg=COLORS["danger"])
        self.status_label.config(text=f"❌ {message}", fg=COLORS["danger"])
        self.is_running = False
        if self.cap:
            self.cap.release()

    def _stop_stream(self):
        """Stop the camera stream"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.status_dot.config(fg=COLORS["text_muted"])
        self.status_label.config(text="Stream stopped", fg=COLORS["text_secondary"])
        
        # Show completion message
        self.info_label.config(text="✓ Stream ended", fg=COLORS["primary"])
        self.after(1000, self.on_done)

    def _cancel(self):
        """Cancel and go back"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.on_cancel()

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()