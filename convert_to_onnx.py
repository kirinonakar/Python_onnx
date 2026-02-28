import os
import torch
import torch.onnx
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

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Export
    if output_path is None:
        output_path = os.path.splitext(model_path)[0] + ".onnx"
    
    dummy_input = torch.randn(1, 3, 64, 64)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'},
                      'output': {2: 'height', 3: 'width'}}
    )
    print(f"Successfully converted to {output_path}")

if __name__ == "__main__":
    # Example usage:
    # convert("RealESRGAN_x4plus.pth", "realesrgan", scale=4)
    # convert("SwinIR_x4.safetensors", "swinir", scale=4)
    pass