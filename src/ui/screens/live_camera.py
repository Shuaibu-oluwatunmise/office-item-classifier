import tkinter as tk
from threading import Thread
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

from components.widgets import header, Card, Button, label, heading, spacer
from components.styles import COLORS, SPACE
from ultralytics import YOLO

class LiveCameraScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, on_done, on_cancel):
        """
        Live camera inference screen with embedded video feed
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

        # Video feed container - LARGE canvas
        video_card = Card(content, shadow="md", padding=SPACE["sm"])
        video_card.pack(fill="both", expand=True)

        # Canvas for video display
        self.canvas = tk.Canvas(
            video_card.outer,
            bg=COLORS["bg_darker"],
            highlightthickness=2,
            highlightbackground=COLORS["border"],
            width=960,
            height=540
        )
        self.canvas.pack(fill="both", expand=True)

        # FPS counter
        self.fps_label = label(
            video_card.outer,
            "FPS: --",
            variant="neon",
            size=10
        )
        self.fps_label.pack(anchor="e", pady=(SPACE["sm"], 0))

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
                    self.font = ImageFont.load_default()
            
            self.after(0, lambda: self.status_label.config(text="Opening camera..."))
            
            # Use camera 1 specifically
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
            
            # Start video loop
            self._video_loop()
            
        except Exception as e:
            self.after(0, lambda: self._camera_error(f"Error: {str(e)}"))

    def _video_loop(self):
        """Main video processing loop"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self._camera_error("Failed to read frame")
            return
        
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
                
                # Draw using PIL for better text
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(frame_pil, "RGBA")
                
                # Text background
                bbox = draw.textbbox((0, 0), label_text, font=self.font if self.font else None)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                pad = 15
                draw.rectangle([(10, 10), (10 + text_w + pad*2, 10 + text_h + pad*2)], 
                             fill=(0, 0, 0, 180))
                draw.text((15, 15), label_text, fill=(255, 255, 255), 
                         font=self.font if self.font else None)
                
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        else:  # detection
            # Draw bounding boxes
            frame = r.plot()
        
        # Convert to PhotoImage for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:  # Canvas is initialized
            img_w, img_h = frame_pil.size
            scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            frame_pil = frame_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(frame_pil)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2 if canvas_w > 1 else 480,
            canvas_h // 2 if canvas_h > 1 else 270,
            image=photo,
            anchor="center"
        )
        self.canvas.image = photo  # Keep reference
        
        # Continue loop
        self.after(1, self._video_loop)

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
        self.status_dot.config(fg=COLORS["text_muted"])
        self.status_label.config(text="Stream stopped", fg=COLORS["text_secondary"])
        
        # Show completion options
        self.after(500, self.on_done)

    def _cancel(self):
        """Cancel and go back"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.on_cancel()

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()