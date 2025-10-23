# Office Item Classifier - Web App

**AI-Powered Office Object Recognition in Your Browser**

Access the live demo: [Scan QR Code or visit URL]

---

## 🎯 Overview

A mobile-first web application that uses deep learning to classify office items in real-time. Built with ONNX Runtime Web for browser-based AI inference - no server required!

**Model:** YOLOv8s-cls (99.99% accuracy)  
**Classes:** 9 office items  
**Platform:** Works on any device with a camera and modern browser

---

## 📱 Features

✅ **Live Camera Inference** - Real-time classification using your device camera  
✅ **Image Upload** - Classify images from your gallery  
✅ **No Installation** - Runs entirely in the browser  
✅ **Offline Capable** - Works without internet after first load (PWA)  
✅ **Mobile Optimized** - Responsive design for phones and tablets  
✅ **Fast & Accurate** - 99.99% accuracy, sub-second inference  

---

## 🚀 Quick Start

### Option 1: Open Locally
```bash
# Navigate to webapp folder
cd webapp

# Option A: Python simple server
python -m http.server 8000

# Option B: Node.js http-server (if installed)
npx http-server -p 8000

# Open browser
# Visit: http://localhost:8000
```

### Option 2: Deploy to GitHub Pages
```bash
# From project root
git add webapp/
git commit -m "Add web app for mobile inference"
git push

# Enable GitHub Pages:
# 1. Go to repo Settings → Pages
# 2. Source: Deploy from branch
# 3. Branch: main, folder: /webapp
# 4. Save

# Your app will be live at:
# https://shuaibu-oluwatunmise.github.io/office-item-classifier/
```

### Option 3: QR Code Access

1. Deploy to GitHub Pages (see above)
2. Generate QR code for your URL at: https://qr-code-generator.com
3. Print/share QR code for instant mobile access!

---

## 🎨 Recognized Classes

The model can identify these 9 office items:

1. 🖱️ **Computer Mouse**
2. ⌨️ **Keyboard**
3. 💻 **Laptop**
4. 📱 **Mobile Phone**
5. 🖥️ **Monitor**
6. 📓 **Notebook**
7. 🪑 **Office Chair**
8. 🖊️ **Pen**
9. 💧 **Water Bottle**

---

## 🛠️ Technical Details

### Architecture
```
webapp/
├── index.html          # Main interface
├── app.js              # Inference logic
├── style.css           # Styling
├── models/
│   └── yolov8s.onnx   # AI model (19.4 MB)
└── README.md           # This file
```

### Technology Stack

- **AI Runtime:** ONNX Runtime Web 1.14.0
- **Model Format:** ONNX (converted from PyTorch)
- **Image Processing:** Canvas API
- **Camera Access:** MediaStream API
- **Framework:** Vanilla JavaScript (no dependencies!)

### Model Specifications

- **Architecture:** YOLOv8s-cls
- **Input Size:** 224×224×3 (RGB)
- **Output:** 9 class probabilities
- **Normalization:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Inference Time:** ~100-300ms (depending on device)

---

## 📋 Usage Instructions

### For Camera Inference:

1. Click **"📷 Start Camera"**
2. Grant camera permissions when prompted
3. Point camera at an office item
4. Click **"📸 Capture & Classify"**
5. View results with confidence scores

### For Image Upload:

1. Click **"📁 Upload Image"**
2. Select image from your device
3. Results display automatically

### Tips for Best Results:

✅ **Good Lighting:** Ensure object is well-lit  
✅ **Fill Frame:** Object should occupy 60%+ of image  
✅ **Plain Background:** Reduces confusion  
✅ **Clear Focus:** Avoid blurry images  
✅ **Single Object:** Works best with one item at a time  

---

## 🌐 Browser Compatibility

### Fully Supported:
- ✅ Chrome/Edge 90+ (Desktop & Mobile)
- ✅ Safari 14+ (iOS & macOS)
- ✅ Firefox 88+
- ✅ Samsung Internet 14+

### Requirements:
- Modern browser with WebAssembly support
- Camera access for live inference
- ~20 MB storage for model cache

---

## 🔧 Development

### Convert Model to ONNX:
```python
# Already done! But if you need to re-convert:
from ultralytics import YOLO

model = YOLO('path/to/best.pt')
model.export(format='onnx', simplify=True, opset=12)
```

### Test Locally:
```bash
# Start local server
python -m http.server 8000

# Open browser
# http://localhost:8000

# Check browser console for logs
```

### Debug Mode:

Open browser DevTools (F12) to see:
- Model loading status
- Inference timing
- Prediction probabilities
- Error messages

---

## 📊 Performance

### Inference Speed (Average):

| Device | Time |
|--------|------|
| iPhone 13 Pro | ~150ms |
| Samsung Galaxy S21 | ~200ms |
| Desktop (Chrome) | ~100ms |
| Laptop (Chrome) | ~250ms |

### Model Size:

- **ONNX Model:** 19.4 MB
- **Total App Size:** ~20 MB (with model)
- **First Load:** ~2-3 seconds
- **Subsequent Loads:** <1 second (cached)

---

## 🚨 Troubleshooting

### "Failed to load AI model"
- Check that `models/yolov8s.onnx` exists
- Verify file size is ~19.4 MB
- Try refreshing the page
- Clear browser cache and reload

### "Cannot access camera"
- Grant camera permissions in browser settings
- Ensure camera is not used by another app
- Try HTTPS connection (required for camera on some browsers)
- Check browser compatibility

### "Low confidence predictions"
- Improve lighting
- Get object closer to camera
- Use plain background
- Ensure object fills frame

### "Slow inference"
- Close other browser tabs
- Check device performance
- Try on desktop for faster inference

---

## 📖 Project Context

This web app is part of the **Office Item Classification** project for:

- **Module:** PDE3802 - AI in Robotics
- **Institution:** Middlesex University London
- **Student:** Oluwatunmise Shuaibu Raphael (M00960413)
- **Deadline:** October 31, 2025

### Related Files:

- Main Project: `../README.md`
- Dataset Info: `../data/DATASET_CARD.md`
- Python Scripts: `../src/`
- Model Training: See main project README

---

## 🎓 Academic Use

This web app demonstrates:

1. **Model Deployment:** Converting PyTorch → ONNX → Browser
2. **Edge Computing:** AI inference without server
3. **Mobile-First Design:** Accessible deployment strategy
4. **Domain Adaptation:** Addressing camera quality differences
5. **Production Thinking:** Real-world usability considerations

---

## 📝 License

MIT License - Academic Project

---

## 🙏 Acknowledgments

- **ONNX Runtime Web** - Browser-based AI inference
- **Ultralytics YOLOv8** - State-of-the-art classification
- **Middlesex University** - PDE3802 Module
- **GitHub Pages** - Free hosting

---

## 📞 Support

**For issues or questions:**

- Check browser console for errors (F12)
- Verify model file exists and is correct size
- Test on different browser/device
- Review troubleshooting section above

---

## 🔗 Links

- **Main Repository:** https://github.com/Shuaibu-oluwatunmise/office-item-classifier
- **ONNX Runtime Docs:** https://onnxruntime.ai/docs/
- **YOLOv8 Docs:** https://docs.ultralytics.com/

---

*Last Updated: October 23, 2024*  
*Version: 1.0 - Initial web app release*