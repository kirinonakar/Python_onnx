import os
import torch
import torch.onnx
import onnx
try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
from basicsr.archs.swinir_arch import SwinIR
from basicsr.archs.rrdbnet_arch import RRDBNet
from safetensors.torch import load_file as load_safetensors

def convert(model_path, arch_type, scale=4, output_path=None):
    print(f"Loading {model_path}...")
    
    # 1. Model Instance
    if arch_type == "realesrgan":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    elif arch_type == "swinir":
        model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                      img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                      num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
    else:
        raise ValueError("Unsupported architecture type. Use 'realesrgan' or 'swinir'.")

    # 2. Load weights
    if model_path.endswith(".safetensors"):
        state_dict = load_safetensors(model_path, device="cpu")
    else:
        state_dict = torch.load(model_path, map_location="cpu")
    
    # Auto-extract params
    for key in ["params_ema", "params", "state_dict", "model"]:
        if key in state_dict:
            state_dict = state_dict[key]
            break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Export
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + ".onnx"
    
    temp_path = output_path + ".temp"
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    print(f"Exporting to {temp_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        temp_path, 
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'},
                      'output': {2: 'height', 3: 'width'}}
    )

    # ONNX Simplifier
    if ONNXSIM_AVAILABLE:
        print("Simplifying ONNX model...")
        try:
            onnx_model = onnx.load(temp_path)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, temp_path)
                print("Simplification successful.")
            else:
                print("Simplification check failed.")
        except Exception as e:
            print(f"Simplification error: {e}")

    print("Merging weights into a single file...")
    try:
        onnx_model = onnx.load(temp_path, load_external_data=True)
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        data_file = temp_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
    except Exception as e:
        if os.path.exists(temp_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
        print(f"Merging failed, kept original: {e}")

    print(f"Successfully converted to {output_path}")

if __name__ == "__main__":
    # Example usage:
    # convert("RealESRGAN_x4plus.pth", "realesrgan", scale=4)
    # convert("SwinIR_x4.safetensors", "swinir", scale=4)
    pass