# 🎨 AI Upscale Model ONNX Converter

A professional suite of tools for converting and optimizing PyTorch image upscaling models to ONNX format.

## ✨ Main Features

- **One-Click Conversion**: Convert `.pth` or `.safetensors` models to ONNX via a modern GUI.
- **Weight Merging**: Merge external weights into a single `.onnx` file for easier deployment.
- **Optimization**: Built-in support for ONNX Simplifier and Opset version selection.

## 🚀 Usage

### Launch the Main Converter
```bash
#activate .venv
source .venv/Scripts/activate
#convert pth or safetensors to onnx
python safe_onnx_converter.py
#convert pth to safetensors
python convert_safetensors.py

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
Real-ESRGAN is working well. Others are work in progress.

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
