# 🎨 AI Upscale Model ONNX Converter

A professional desktop application for converting PyTorch image upscaling models (Real-ESRGAN, SwinIR, Real-HAT-GAN) to ONNX format with optimized settings.

## ✨ Features

- **One-Click Conversion**: Convert `.pth` or `.safetensors` models to ONNX with a single click.
- **Multiple Architectures**: Supports:
  - Real-ESRGAN (23B, Anime, Light)
  - SwinIR (Classic, Light, Large)
  - Real-HAT-GAN (HAT-L, HAT-Small)
- **Smart Optimization**: Automatically detects model type and applies optimal export settings.
- **Dynamic Input Sizing**: Automatically calculates dummy input sizes based on model architecture.
- **Professional UI**: Modern dark theme with gradient effects and smooth animations.
- **Status Tracking**: Real-time status updates and error reporting.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Custom Dependencies (basicsr, safetensors, etc.)

### Installation

1.  **Clone the repository** (or download the script).

2.  **Install dependencies**:
    ```bash
    pip install torch customtkinter safetensors
    # Install additional requirements if needed
    pip install -r requirements.txt
    ```

## 🛠️ Usage

1.  **Launch the application**:
    ```bash
    python gui_converter.py
    ```

2.  **Select Model File**:
    - Click **Browse** and select your `.pth` or `.safetensors` model file.

3.  **Configure Settings**:
    - **Architecture**: Select the model architecture (e.g., "Real-ESRGAN (23B)").
    - **Scale**: Choose the upscaling factor (2, 3, 4, or 8).
    - **Window Size**: (Optional) Adjust for SwinIR models (default: 8).

4.  **Convert**:
    - Click the **CONVERT TO ONNX** button.
    - The application will automatically detect the model type and export it to ONNX format in the same directory as the source model.

## 📂 Project Structure

```
Python_onnx/
├── gui_converter.py          # Main application window and logic
├── hat_arch.py               # HAT model architecture (if needed)
├── basicsr/                  # Required libraries (if included)
│   └── ...
└── README.md                 # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.
