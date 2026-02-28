import os
import torch
import torch.onnx
import customtkinter as ctk
import warnings
import traceback
from tkinter import filedialog, messagebox
from safetensors.torch import load_file as load_safetensors

# Meshgrid warning suppression
warnings.filterwarnings("ignore", category=UserWarning)

# Import common architectures
try:
    from basicsr.archs.swinir_arch import SwinIR
    from basicsr.archs.rrdbnet_arch import RRDBNet
    BASI_SR_AVAILABLE = True
except ImportError:
    BASI_SR_AVAILABLE = False
    SwinIR = None
    RRDBNet = None

# Import HAT architecture
try:
    from hat_arch import HAT
    HAT_AVAILABLE = True
except ImportError:
    HAT_AVAILABLE = False
    HAT = None

# Import Real-CUGAN architecture
try:
    from cugan_arch import RealCUGAN
    CUGAN_AVAILABLE = True
except ImportError:
    CUGAN_AVAILABLE = False
    RealCUGAN = None

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ONNXConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Upscale Model ONNX Converter")
        self.geometry("700x600")
        self.grid_columnconfigure(0, weight=1)

        # Header with Gradient-like effect
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="#1f2937")
        self.header_frame.pack(fill="x", pady=(0, 20))
        
        self.label = ctk.CTkLabel(self.header_frame, text="✨ Image Upscale to ONNX ✨", font=("Orbitron", 24, "bold"), text_color="#60a5fa")
        self.label.pack(pady=20)

        # Main Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=40)

        # 1. Model File Selection
        self.file_label = ctk.CTkLabel(self.main_container, text="Model File (.pth / .safetensors)", font=("Arial", 14, "bold"))
        self.file_label.pack(anchor="w", pady=(10, 5))
        
        self.file_frame = ctk.CTkFrame(self.main_container, fg_color="#374151")
        self.file_frame.pack(fill="x", pady=5)

        self.model_path_entry = ctk.CTkEntry(self.file_frame, placeholder_text="Path to your model...", border_width=0, height=35)
        self.model_path_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        self.browse_btn = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_file, width=100, fg_color="#3b82f6", hover_color="#2563eb")
        self.browse_btn.pack(side="right", padx=10, pady=10)

        # 2. Architecture and Scale Selection
        self.settings_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.settings_frame.pack(fill="x", pady=10)

        # Architecture
        self.arch_col = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.arch_col.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        self.arch_label = ctk.CTkLabel(self.arch_col, text="Architecture", font=("Arial", 14, "bold"))
        self.arch_label.pack(anchor="w", pady=5)
        
        self.arch_var = ctk.StringVar(value="Real-ESRGAN (23B)")
        self.arch_dropdown = ctk.CTkOptionMenu(self.arch_col, 
                                             values=[
                                                 "Real-ESRGAN (23B)", 
                                                 "Real-ESRGAN Anime (6B)", 
                                                 "Real-ESRGAN Light (8B)", 
                                                 "SwinIR (Large - RealSR)",
                                                 "SwinIR (Classic - RealSR)",
                                                 "SwinIR (Classic)",
                                                 "SwinIR (Light)",
                                                 "SwinIR (Large)",
                                                 "Real-HAT-GAN", 
                                                 "HAT-L",
                                                 "HAT-Small",
                                                 "Real-CUGAN 2X/4X",
                                                 "Real-CUGAN 3X"
                                             ], 
                                             variable=self.arch_var, fg_color="#374151", button_color="#3b82f6")
        self.arch_dropdown.pack(fill="x")

        # Upscale Factor
        self.scale_col = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.scale_col.pack(side="left", padx=(10, 0))
        
        self.scale_label = ctk.CTkLabel(self.scale_col, text="Scale", font=("Arial", 14, "bold"))
        self.scale_label.pack(anchor="w", pady=5)
        
        self.scale_var = ctk.StringVar(value="4")
        self.scale_dropdown = ctk.CTkOptionMenu(self.scale_col, values=["2", "3", "4", "8"], variable=self.scale_var, width=100, fg_color="#374151", button_color="#3b82f6")
        self.scale_dropdown.pack()

        # 3. Optimization Parameters
        self.opt_label = ctk.CTkLabel(self.main_container, text="Export Settings", font=("Arial", 14, "bold"))
        self.opt_label.pack(anchor="w", pady=(20, 5))

        self.opt_frame = ctk.CTkFrame(self.main_container, fg_color="#374151")
        self.opt_frame.pack(fill="x", pady=5)

        self.input_size_label = ctk.CTkLabel(self.opt_frame, text="Dummy Input Size (H, W):")
        self.input_size_label.pack(side="left", padx=15, pady=15)
        
        self.input_size_entry = ctk.CTkEntry(self.opt_frame, placeholder_text="64, 64", width=80)
        self.input_size_entry.insert(0, "64,64")
        self.input_size_entry.pack(side="left", padx=5)

        self.win_label = ctk.CTkLabel(self.opt_frame, text="Window Size:")
        self.win_label.pack(side="left", padx=(15, 5))
        self.win_var = ctk.StringVar(value="8")
        self.win_entry = ctk.CTkEntry(self.opt_frame, placeholder_text="8 or 16", width=50)
        self.win_entry.insert(0, "8")
        self.win_entry.pack(side="left", padx=5)

        # 4. Convert Button
        self.convert_btn = ctk.CTkButton(self, text="🚀 CONVERT TO ONNX", command=self.convert_model, height=50, font=("Arial", 18, "bold"), fg_color="#10b981", hover_color="#059669")
        self.convert_btn.pack(pady=40, padx=40, fill="x")

        # Status Footer
        self.status_footer = ctk.CTkFrame(self, height=40, fg_color="#111827", corner_radius=0)
        self.status_footer.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(self.status_footer, text="System Ready", text_color="#9ca3af", font=("Arial", 12))
        self.status_label.pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth *.safetensors"), ("PyTorch", "*.pth"), ("Safetensors", "*.safetensors")])
        if file_path:
            self.model_path_entry.delete(0, "end")
            self.model_path_entry.insert(0, file_path)

    def get_model_instance(self, arch, scale, window_size=8, img_size=64):
        if "Real-ESRGAN" in arch:
            if not BASI_SR_AVAILABLE:
                raise ImportError("basicsr not found. Please install it with 'pip install basicsr'")
            if arch == "Real-ESRGAN (23B)":
                return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            elif arch == "Real-ESRGAN Anime (6B)":
                return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
            elif arch == "Real-ESRGAN Light (8B)":
                return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=8, num_grow_ch=32, scale=scale)
        
        elif "SwinIR" in arch:
            if not BASI_SR_AVAILABLE:
                raise ImportError("basicsr not found. Please install it with 'pip install basicsr'")
            if arch == "SwinIR (Large - RealSR)":
                # Large Real-SwinIR (Common for Real-ESRGAN/Real-SR)
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240, 
                             num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8], mlp_ratio=2, 
                             upsampler='nearest+conv', resi_connection='3conv')
            elif arch == "SwinIR (Classic - RealSR)":
                # Medium Real-SwinIR
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                             num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, 
                             upsampler='nearest+conv', resi_connection='3conv')
            elif arch == "SwinIR (Classic)":
                # Regular SwinIR
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                             num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
            elif arch == "SwinIR (Light)":
                # Lightweight SwinIR
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60, 
                             num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
            elif arch == "SwinIR (Large)":
                # Large SwinIR
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240, 
                             num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8], mlp_ratio=2, upsampler='pixelshuffle')
        
        elif "HAT" in arch:
            if not HAT_AVAILABLE:
                raise ImportError("HAT architecture code (hat_arch.py) not found or einops missing.")
            
            if arch == "Real-HAT-GAN":
                return HAT(img_size=img_size, patch_size=1, in_chans=3, embed_dim=180, 
                          depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6), 
                          window_size=window_size, compress_ratio=3, squeeze_factor=30, 
                          conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2., 
                          upscale=scale, upsampler='pixelshuffle', resi_connection='1conv')
            elif arch == "HAT-L":
                return HAT(img_size=img_size, patch_size=1, in_chans=3, embed_dim=180, 
                          depths=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6), 
                          window_size=window_size, compress_ratio=3, squeeze_factor=30, 
                          conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2., 
                          upscale=scale, upsampler='pixelshuffle', resi_connection='1conv')
            elif arch == "HAT-Small":
                return HAT(img_size=img_size, patch_size=1, in_chans=3, embed_dim=144, 
                          depths=(6, 6, 6, 6, 6, 6), num_heads=(6, 6, 6, 6, 6, 6), 
                          window_size=window_size, compress_ratio=3, squeeze_factor=30, 
                          conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2., 
                          upscale=scale, upsampler='pixelshuffle', resi_connection='1conv')
        
        elif "Real-CUGAN" in arch:
            if not CUGAN_AVAILABLE:
                raise ImportError("Real-CUGAN architecture code (cugan_arch.py) not found.")
            # Real-CUGAN 2x and 4x use the same UNet1/UNet2 structure with different deconvs in the base class
            # This simplified RealCUGAN class handles it via the initialization
            return RealCUGAN(in_channels=3, out_channels=3, scale=scale)

        return None

    def convert_model(self):
        model_path = self.model_path_entry.get().strip()
        arch = self.arch_var.get()
        scale = int(self.scale_var.get())
        input_size_str = self.input_size_entry.get().strip()
        try:
            window_size = int(self.win_entry.get().strip())
        except ValueError:
            window_size = 8

        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Selection Error", "Please select a valid model file.")
            return

        try:
            h, w = map(int, input_size_str.split(","))
        except ValueError:
            messagebox.showerror("Format Error", "Invalid input size. Use format: H, W (e.g., 64, 64)")
            return

        self.status_label.configure(text="PROCESSING... PLEASE WAIT", text_color="#fbbf24")
        self.convert_btn.configure(state="disabled")
        self.update()

        try:
            # 1. Initialize Model
            model = self.get_model_instance(arch, scale, window_size, img_size=h)
            if model is None:
                raise ValueError(f"Unsupported architecture selected: {arch}")

            # 2. Load weights
            if model_path.endswith(".safetensors"):
                state_dict = load_safetensors(model_path, device="cpu")
            else:
                state_dict = torch.load(model_path, map_location="cpu")
            
            # Handle BasicsSR / Auto-discovery of the inner state dict
            for key in ["params_ema", "params", "state_dict", "model"]:
                if key in state_dict:
                    state_dict = state_dict[key]
                    break

            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Load weights
            model.load_state_dict(new_state_dict, strict=True)
            model.eval()

            # 3. Create dummy input
            dummy_input = torch.randn(1, 3, h, w)

            # 4. Export
            output_onnx = os.path.splitext(model_path)[0] + ".onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                output_onnx,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {2: 'height', 3: 'width'},
                    'output': {2: 'height', 3: 'width'}
                }
            )

            self.status_label.configure(text=f"SUCCESS: {os.path.basename(output_onnx)}", text_color="#34d399")
            messagebox.showinfo("Export Successful", f"Model converted successfully!\n\nTarget: {output_onnx}")

        except Exception as e:
            error_msg = str(e)
            full_traceback = traceback.format_exc()
            
            # Save to log file
            with open("error_log.txt", "w", encoding="utf-8") as f:
                f.write(f"Error Message: {error_msg}\n")
                f.write("-" * 50 + "\n")
                f.write(full_traceback)
            
            self.status_label.configure(text="CONVERSION FAILED - LOG SAVED", text_color="#f87171")
            messagebox.showerror("Conversion Process Error", 
                                 f"Error: {error_msg}\n\nFull log saved to error_log.txt")
        finally:
            self.convert_btn.configure(state="normal")

if __name__ == "__main__":
    app = ONNXConverterApp()
    app.mainloop()
