# 🎨 AI Upscale Model ONNX Converter

A professional suite of tools for converting and optimizing PyTorch image upscaling models (Real-ESRGAN, SwinIR, Real-HAT-GAN, Real-CUGAN) to ONNX format.

## ✨ Main Features

- **One-Click Conversion**: Convert `.pth` or `.safetensors` models to ONNX via a modern GUI.
- **Auto-Detection**: Intelligent window size detection for HAT and SwinIR models.
- **Weight Merging**: Merge external weights into a single `.onnx` file for easier deployment.
- **Optimization**: Built-in support for ONNX Simplifier and Opset version selection.
- **Multiple Architectures**: Supports HAT, SwinIR, Real-ESRGAN, and Real-CUGAN.

## 🚀 Usage

### Launch the Main Converter
```bash
python safe_onnx_converter.py
```
This is the **primary tool** recommended for most users. It provides the most features and the best compatibility.

---

## � File Descriptions

### 🖼️ Core Converters
- **`safe_onnx_converter.py`**: **[Primary]** The most advanced GUI converter. Features include auto-detection of parameters, Opset selection (11-18), simplification, and weight merging.
- **`convert_safetensors.py`**: A dedicated GUI for converting PyTorch `.pth` files to the newer `.safetensors` format.


### 🛠️ Optimization & Utilities
- **`convert_int32.py`**: Converts ONNX models from Int64 to Int32. This is often required for environments like Rust (ort crate) or mobile deployment to avoid data type errors.
- **`check_sd.py`**: A diagnostic tool to inspect the internal structure (State Dict) of model files to verify layers and weights.

### 📐 Architecture Definitions
These files provide the model logic required during the conversion process:
- **`hat_arch.py`**: Hybrid Attention Transformer (HAT) architecture. Modified to prevent `Loop` operator errors in ONNX.
- **`cugan_arch.py`**: Real-CUGAN architecture definition.

---

## 💡 Troubleshooting Tips

### HAT Model Conversion Errors
- HAT model conversion errors are due to the fact that the HAT model is not compatible with the latest version of ONNX. 
---

## 📦 Installation

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install torch torchvision onnx onnxsim customtkinter safetensors
    # Or use the requirements file
    pip install -r requirements.txt
    ```

## 🤝 Contributing

Feel free to fork the repository and submit pull requests for new architectures or optimizations!
