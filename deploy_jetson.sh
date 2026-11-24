#!/bin/bash
# Jetson deployment script — run on the Jetson device.
#
# Sets up the full cry-agent environment on Jetson Orin Nano/NX
# with JetPack 6+.

set -e

echo "=========================================="
echo "Cry-Agent Jetson Deployment"
echo "=========================================="

# Check we're on Jetson
if [ ! -d /usr/local/cuda ]; then
    echo "WARNING: CUDA not found. Are you on a Jetson device?"
    echo "Continue anyway? (y/n)"
    read -r response
    [ "$response" != "y" ] && exit 1
fi

# 1. System dependencies
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv python3-dev \
    libsndfile1 libsndfile1-dev \
    libportaudio2 portaudio19-dev \
    cmake build-essential \
    v4l-utils  # for camera tools

# 2. Python virtual environment
echo "[2/6] Creating Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# 3. Python packages
echo "[3/6] Installing Python packages..."
pip install \
    librosa soundfile scikit-learn joblib \
    imbalanced-learn \
    sounddevice \
    numpy scipy

# MediaPipe (CPU mode on Jetson — GPU acceleration via TensorRT)
echo "[4/6] Installing MediaPipe + OpenCV..."
pip install opencv-python mediapipe || {
    echo "MediaPipe install failed — trying alternative..."
    # On Jetson, mediapipe may need special build
    pip install opencv-python
    echo "MediaPipe not available — will use annotation fallback"
}

# 4. Copy model files
echo "[5/6] Checking model files..."
if [ ! -f ~/.config/cry-agent/models/cry_classifier.joblib ]; then
    echo "Model not found. Training from cached datasets..."
    # If datasets are cached, retrain
    python3 train_combined_v2.py || {
        echo "Cannot train — datasets not available."
        echo "Copy model from dev machine:"
        echo "  scp ~/.config/cry-agent/models/*.joblib jetson@<IP>:~/.config/cry-agent/models/"
    }
else
    echo "Model found."
fi

# 5. Camera check
echo "[6/6] Checking camera..."
if command -v v4l2-ctl &> /dev/null; then
    echo "Available video devices:"
    v4l2-ctl --list-devices 2>/dev/null || echo "  (no devices found)"
fi

# 6. Microphone check
echo "Checking audio devices:"
python3 -c "
try:
    import sounddevice as sd
    devices = sd.query_devices()
    inputs = [d for d in devices if d['max_input_channels'] > 0]
    if inputs:
        print(f'  Found {len(inputs)} input device(s):')
        for i, d in enumerate(inputs):
            print(f'    [{i}] {d[\"name\"]}')
    else:
        print('  No input devices found')
except Exception as e:
    print(f'  Audio check failed: {e}')
"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Quick test (replay mode, no camera/mic needed):"
echo "  python3 live_camera.py --replay-audio <wav_file> --postures rooting"
echo ""
echo "Full live mode:"
echo "  python3 live_camera.py --camera 0 --mic-device 0 --temp 22"
echo ""
echo "Check camera:"
echo "  python3 -c \"import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')\""
