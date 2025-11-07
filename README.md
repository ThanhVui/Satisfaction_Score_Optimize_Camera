# Satisfaction Score Camera Analysis

## Overview
This project is a web-based application that analyzes satisfaction scores through emotion recognition using camera input. It utilizes deep learning (PyTorch) and computer vision techniques to detect and analyze facial expressions in real-time.

## Features
- Real-time emotion recognition through webcam
- Video file upload and analysis
- Satisfaction score calculation based on facial expressions
- Web interface for easy interaction
- Support for both live camera feed and video file processing

## Project Structure
```
├── app.py              # Main Flask application
├── docker-compose.yml  # Docker compose configuration
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
├── images/           # Image assets
├── models/           # Pre-trained models
│   └── best_model_resemotenet_80.pth
├── static/           # Static files
│   └── uploads/     # Upload directory for videos
├── templates/        # HTML templates
│   ├── camera.html
│   ├── index.html
│   ├── result.html
│   ├── video_stream.html
│   └── video_upload.html
└── train_models/    # Jupyter notebooks for model training
    ├── emotion-to-satisfaction-01.ipynb
    └── resemotenet-rafdb-fer-affectnet-26-final.ipynb
```

## Prerequisites

### System Requirements
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Webcam (for live analysis)
- RAM: Minimum 8GB (16GB recommended)
- Storage: At least 2GB free space

### Key Dependencies
- Flask 3.1.2
- PyTorch 2.8.0
- TorchVision 0.23.0
- OpenCV (opencv-python) 4.12.0.88
- Mediapipe 0.10.14
- NumPy 2.2.6
- Pillow 11.3.0

### Operating System Support
- Windows 10/11
- Linux (Ubuntu 20.04 or higher)
- macOS (10.15 or higher)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ThanhVui/Satisfaction_Score_Optimize_Camera.git
cd Satisfaction_Score_Optimize_Camera
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Method 1: Using Python directly

1. Ensure you're in the project directory and your virtual environment is activated:
```bash
# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

2. Set up environment variables (if needed):
```bash
# On Windows PowerShell
$env:FLASK_ENV = "development"
$env:FLASK_APP = "app.py"

# On Linux/Mac
export FLASK_ENV=development
export FLASK_APP=app.py
```

3. Launch the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Method 2: Using Docker

1. Ensure Docker and Docker Compose are installed on your system

2. Build and run the containers:
```bash
docker-compose up --build
```

3. For subsequent runs (if no changes to requirements):
```bash
docker-compose up
```

4. To stop the application:
```bash
docker-compose down
```

### Troubleshooting

If you encounter any issues:

1. GPU-related errors:
   - Ensure CUDA is properly installed
   - Check if PyTorch is using GPU: Run `python -c "import torch; print(torch.cuda.is_available())"`
   - Update GPU drivers if necessary

2. Webcam access issues:
   - Check camera permissions in your browser
   - Ensure no other application is using the camera
   - Try refreshing the page

3. Package conflicts:
   - Delete the virtual environment and create a new one
   - Install requirements with `pip install -r requirements.txt --no-cache-dir`

## Usage
1. Open your web browser and navigate to `http://localhost:5000`
2. Choose between:
   - Live camera analysis
   - Video file upload
3. View the satisfaction scores and emotion analysis results

## Model Information
The project uses ResEmoteNet, a deep learning model trained on multiple emotion recognition datasets including RAF-DB, FER, and AffectNet. The model is optimized for real-time emotion detection and satisfaction score calculation.

## License
[Your License Here]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
- GitHub: [@ThanhVui](https://github.com/ThanhVui)