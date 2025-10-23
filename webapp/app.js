// ========================================
// Office Item Classifier - Main App Logic
// UPDATED: Live inference + image preview
// ========================================

// Class names (must match training order)
const CLASS_NAMES = [
    'Computer Mouse',
    'Keyboard', 
    'Laptop',
    'Mobile Phone',
    'Monitor',
    'Notebook',
    'Office Chair',
    'Pen',
    'Water Bottle'
];

// Global variables
let session = null;
let stream = null;
let isModelLoaded = false;
let isInferencing = false;
let animationFrameId = null;

// DOM Elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const fileInput = document.getElementById('fileInput');
const loadingDiv = document.getElementById('loading');
const predictionDiv = document.getElementById('prediction');
const resultsDiv = document.getElementById('results');

// ========================================
// Initialize App
// ========================================

window.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Office Item Classifier loading...');
    await loadModel();
});

// ========================================
// Model Loading
// ========================================

async function loadModel() {
    try {
        console.log('📦 Loading ONNX model...');
        showLoading('Loading AI model...');

        // Configure ONNX Runtime
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';
        
        // Load the model
        session = await ort.InferenceSession.create('models/yolov8s.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        isModelLoaded = true;
        hideLoading();
        console.log('✅ Model loaded successfully!');
        showToast('Model loaded! Ready to classify.', 'success');

    } catch (error) {
        console.error('❌ Error loading model:', error);
        hideLoading();
        showError('Failed to load AI model. Please refresh the page.');
    }
}

// ========================================
// Camera Control - LIVE INFERENCE
// ========================================

startBtn.addEventListener('click', async () => {
    if (!isModelLoaded) {
        showError('Model is still loading. Please wait...');
        return;
    }

    try {
        console.log('📷 Starting camera with LIVE inference...');
        
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment', // Use back camera on mobile
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        webcamElement.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => {
                webcamElement.play();
                resolve();
            };
        });

        // Update button states
        startBtn.disabled = true;
        stopBtn.disabled = false;

        // Start live inference loop
        isInferencing = true;
        runLiveInference();

        console.log('✅ Camera started - Live classification active!');
        showToast('Live classification active!', 'success');

    } catch (error) {
        console.error('❌ Camera error:', error);
        showError('Cannot access camera. Please grant camera permissions.');
    }
});

stopBtn.addEventListener('click', () => {
    console.log('⏸️ Stopping camera...');
    stopCamera();
});

function stopCamera() {
    isInferencing = false;
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcamElement.srcObject = null;
        stream = null;
    }

    // Update button states
    startBtn.disabled = false;
    stopBtn.disabled = true;

    console.log('✅ Camera stopped');
}

// ========================================
// Live Inference Loop
// ========================================

async function runLiveInference() {
    if (!isInferencing) return;

    try {
        // Capture current frame
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        const context = canvasElement.getContext('2d');
        context.drawImage(webcamElement, 0, 0);

        // Get image data
        const imageData = context.getImageData(0, 0, canvasElement.width, canvasElement.height);
        
        // Classify (without showing loading spinner for smooth UX)
        await classifyImage(imageData, true); // true = live mode

    } catch (error) {
        console.error('❌ Live inference error:', error);
    }

    // Continue loop (aim for ~2 FPS to not overwhelm)
    setTimeout(() => {
        animationFrameId = requestAnimationFrame(runLiveInference);
    }, 500); // Classify every 500ms
}

// ========================================
// File Upload - WITH IMAGE PREVIEW
// ========================================

fileInput.addEventListener('change', async (event) => {
    if (!isModelLoaded) {
        showError('Model is still loading. Please wait...');
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    console.log('📁 Processing uploaded image...');

    try {
        // Load image
        const img = await loadImageFromFile(file);
        
        // Show image preview in video container
        showImagePreview(img);

        // Draw to canvas
        canvasElement.width = img.width;
        canvasElement.height = img.height;
        const context = canvasElement.getContext('2d');
        context.drawImage(img, 0, 0);

        // Get image data
        const imageData = context.getImageData(0, 0, canvasElement.width, canvasElement.height);
        
        // Classify
        await classifyImage(imageData, false); // false = not live mode

    } catch (error) {
        console.error('❌ Error processing image:', error);
        showError('Failed to process image. Please try another image.');
    }
});

function showImagePreview(img) {
    // Hide video, show image
    webcamElement.style.display = 'none';
    
    // Create or get preview element
    let preview = document.getElementById('imagePreview');
    if (!preview) {
        preview = document.createElement('img');
        preview.id = 'imagePreview';
        preview.style.width = '100%';
        preview.style.height = '100%';
        preview.style.objectFit = 'contain';
        webcamElement.parentElement.appendChild(preview);
    }

    preview.src = img.src;
    preview.style.display = 'block';
}

function hideImagePreview() {
    const preview = document.getElementById('imagePreview');
    if (preview) {
        preview.style.display = 'none';
    }
    webcamElement.style.display = 'block';
}

function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ========================================
// Image Classification
// ========================================

async function classifyImage(imageData, isLive = false) {
    try {
        if (!isLive) {
            showLoading('Analyzing image...');
        }

        // Preprocess image
        const input = preprocessImage(imageData);

        // Run inference
        const feeds = { images: input };
        const results = await session.run(feeds);

        // Get output tensor
        const output = results[Object.keys(results)[0]];
        const predictions = output.data;

        // Get top 3 predictions
        const top3 = getTop3Predictions(predictions);

        // Display results
        displayPrediction(top3, isLive);

        if (!isLive) {
            hideLoading();
        }

    } catch (error) {
        console.error('❌ Classification error:', error);
        if (!isLive) {
            hideLoading();
            showError('Classification failed. Please try again.');
        }
    }
}

// ========================================
// Image Preprocessing
// ========================================

function preprocessImage(imageData) {
    const { data, width, height } = imageData;

    // Resize to 224x224
    const targetSize = 224;
    const resized = resizeImage(data, width, height, targetSize, targetSize);

    // Convert to tensor format [1, 3, 224, 224]
    const float32Data = new Float32Array(3 * targetSize * targetSize);
    
    // Normalize and convert to CHW format
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < targetSize * targetSize; i++) {
        const r = resized[i * 4] / 255;
        const g = resized[i * 4 + 1] / 255;
        const b = resized[i * 4 + 2] / 255;

        float32Data[i] = (r - mean[0]) / std[0];
        float32Data[targetSize * targetSize + i] = (g - mean[1]) / std[1];
        float32Data[2 * targetSize * targetSize + i] = (b - mean[2]) / std[2];
    }

    return new ort.Tensor('float32', float32Data, [1, 3, targetSize, targetSize]);
}

function resizeImage(data, width, height, newWidth, newHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = newWidth;
    canvas.height = newHeight;
    const ctx = canvas.getContext('2d');

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    const imageData = new ImageData(new Uint8ClampedArray(data), width, height);
    tempCtx.putImageData(imageData, 0, 0);

    ctx.drawImage(tempCanvas, 0, 0, newWidth, newHeight);

    return ctx.getImageData(0, 0, newWidth, newHeight).data;
}

// ========================================
// Results Processing
// ========================================

function getTop3Predictions(predictions) {
    const probs = Array.from(predictions).map((prob, idx) => ({
        class: CLASS_NAMES[idx],
        confidence: prob,
        index: idx
    }));

    probs.sort((a, b) => b.confidence - a.confidence);

    return probs.slice(0, 3);
}

// ========================================
// Display Results
// ========================================

function displayPrediction(top3, isLive = false) {
    // Show prediction section
    predictionDiv.style.display = 'block';
    resultsDiv.querySelector('.result-placeholder')?.remove();

    // Display top prediction
    const topPrediction = top3[0];
    document.getElementById('className').textContent = topPrediction.class;
    document.getElementById('confidence').textContent = `${(topPrediction.confidence * 100).toFixed(2)}%`;

    // Update confidence bar
    const confidenceBar = document.getElementById('confidenceBarFill');
    confidenceBar.style.width = `${topPrediction.confidence * 100}%`;

    // Color based on confidence
    if (topPrediction.confidence > 0.9) {
        confidenceBar.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
    } else if (topPrediction.confidence > 0.7) {
        confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
    } else {
        confidenceBar.style.background = 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)';
    }

    // Display top 3
    const top3List = document.getElementById('top3List');
    top3List.innerHTML = '';

    top3.forEach((pred, idx) => {
        const item = document.createElement('div');
        item.className = 'top3-item';
        item.innerHTML = `
            <span class="top3-rank">${idx + 1}.</span>
            <span class="top3-class">${pred.class}</span>
            <span class="top3-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
        `;
        top3List.appendChild(item);
    });

    if (!isLive) {
        console.log('📊 Top prediction:', topPrediction.class, `(${(topPrediction.confidence * 100).toFixed(2)}%)`);
    }
}

// ========================================
// UI Helpers
// ========================================

function showLoading(message = 'Loading...') {
    loadingDiv.classList.remove('hidden');
    loadingDiv.querySelector('p').textContent = message;
}

function hideLoading() {
    loadingDiv.classList.add('hidden');
}

function showError(message) {
    const errorToast = document.getElementById('errorToast');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorToast.style.display = 'block';

    setTimeout(() => {
        errorToast.style.display = 'none';
    }, 5000);
}

function showToast(message, type = 'success') {
    console.log(`📢 ${type.toUpperCase()}: ${message}`);
}

// ========================================
// Cleanup
// ========================================

window.addEventListener('beforeunload', () => {
    stopCamera();
});

console.log('✅ App initialized - Live inference mode ready!');