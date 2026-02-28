import torch
from safetensors.torch import load_file as load_safetensors

def check_weights(path):
    print(f"Checking {path}...")
    try:
        if path.endswith(".safetensors"):
            sd = load_safetensors(path, device="cpu")
        else:
            sd = torch.load(path, map_location="cpu")
        
        if "params" in sd: sd = sd["params"]
        if "state_dict" in sd: sd = sd["state_dict"]
        
        for k in ["unet1.conv_bottom.weight", "unet2.conv_bottom.weight", "conv_final.weight"]:
            if k in sd:
                print(f"{k}: {sd[k].shape}")
            else:
                print(f"{k}: NOT FOUND")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # I don't know the exact path yet, but I can try to find it from the GUI log or common places.
    # Actually, the user just ran gui_converter.py, so I can't easily know the path.
    # I'll just assume the user can tell me or I'll look at the error_log.txt.
    pass
