import os
import torch
import torch.onnx
import onnx
import customtkinter as ctk
import warnings
import traceback
from tkinter import filedialog, messagebox
from safetensors.torch import load_file as load_safetensors

# Meshgrid 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 아키텍처 임포트
try:
    from basicsr.archs.swinir_arch import SwinIR
    from basicsr.archs.rrdbnet_arch import RRDBNet
    BASI_SR_AVAILABLE = True
except ImportError:
    BASI_SR_AVAILABLE = False
    SwinIR = None
    RRDBNet = None

# HAT 아키텍처 임포트 (hat_arch.py 필요)
try:
    from hat_arch import HAT
    HAT_AVAILABLE = True
except ImportError:
    HAT_AVAILABLE = False
    HAT = None

# Real-CUGAN 아키텍처 임포트 (cugan_arch.py 필요)
try:
    from cugan_arch import RealCUGAN
    CUGAN_AVAILABLE = True
except ImportError:
    CUGAN_AVAILABLE = False
    RealCUGAN = None

# ONNX Simplifier 임포트
try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    simplify = None

# UI 설정
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SafeToONNXConverter(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Safetensors to ONNX Converter Pro")
        self.geometry("750x700")
        self.grid_columnconfigure(0, weight=1)

        # 헤더 섹션
        self.header_frame = ctk.CTkFrame(self, height=100, corner_radius=10, fg_color="#1e293b")
        self.header_frame.pack(fill="x", pady=10, padx=20)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="✨ AI Model ONNX Converter ✨", 
            font=("Segoe UI", 26, "bold"), 
            text_color="#60a5fa"
        )
        self.title_label.pack(pady=(20, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame, 
            text="Safetensors / PTH to ONNX with Simplification", 
            font=("Segoe UI", 14), 
            text_color="#94a3b8"
        )
        self.subtitle_label.pack(pady=(0, 15))

        # 메인 컨테이너
        self.content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # 1. 파일 선택 섹션
        self.create_section_label(self.content_frame, "1. 모델 파일 선택 (.safetensors / .pth)")
        self.file_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.file_frame.pack(fill="x", pady=(5, 20), padx=5)

        self.path_entry = ctk.CTkEntry(
            self.file_frame, 
            placeholder_text="모델 파일을 선택하세요...", 
            border_width=0, 
            height=40,
            fg_color="#1e293b"
        )
        self.path_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        self.browse_btn = ctk.CTkButton(
            self.file_frame, 
            text="파일 열기", 
            command=self.browse_file, 
            width=100, 
            height=40,
            fg_color="#3b82f6", 
            hover_color="#2563eb"
        )
        self.browse_btn.pack(side="right", padx=10, pady=10)

        # 2. 아키텍처 및 설정 섹션
        self.create_section_label(self.content_frame, "2. 모델 아키텍처 및 파라미터")
        self.settings_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.settings_frame.pack(fill="x", pady=(5, 20), padx=5)

        # Architecture Dropdown
        self.arch_label = ctk.CTkLabel(self.settings_frame, text="아키텍처:", font=("Segoe UI", 13, "bold"))
        self.arch_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.arch_var = ctk.StringVar(value="Real-HAT-GAN")
        self.arch_menu = ctk.CTkOptionMenu(
            self.settings_frame, 
            values=[
                "Real-HAT-GAN", "HAT-L", "HAT-Small",
                "Real-ESRGAN (23B)", "Real-ESRGAN Anime (6B)", "Real-ESRGAN Light (8B)",
                "SwinIR (Large - RealSR)", "SwinIR (Classic)", "SwinIR (Light)",
                "Real-CUGAN 2X/4X", "Real-CUGAN 3X"
            ],
            variable=self.arch_var,
            width=250,
            fg_color="#1e293b",
            button_color="#3b82f6"
        )
        self.arch_menu.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="w")

        # Scale Selection
        self.scale_label = ctk.CTkLabel(self.settings_frame, text="배율 (Scale):", font=("Segoe UI", 13, "bold"))
        self.scale_label.grid(row=0, column=1, padx=15, pady=(15, 5), sticky="w")
        
        self.scale_var = ctk.StringVar(value="4")
        self.scale_menu = ctk.CTkOptionMenu(
            self.settings_frame, 
            values=["2", "3", "4", "8"], 
            variable=self.scale_var,
            width=100,
            fg_color="#1e293b",
            button_color="#3b82f6"
        )
        self.scale_menu.grid(row=1, column=1, padx=15, pady=(0, 15), sticky="w")

        # 3. 상세 내보내기 설정
        self.create_section_label(self.content_frame, "3. ONNX 내보내기 세부 설정")
        self.export_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.export_frame.pack(fill="x", pady=(5, 20), padx=5)

        # Input Size
        self.size_label = ctk.CTkLabel(self.export_frame, text="더미 입력 크기 (H,W):", font=("Segoe UI", 12))
        self.size_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
        self.size_entry = ctk.CTkEntry(self.export_frame, width=100, fg_color="#1e293b")
        self.size_entry.insert(0, "256,256")
        self.size_entry.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        # Window Size
        self.win_label = ctk.CTkLabel(self.export_frame, text="윈도우 크기 (Window):", font=("Segoe UI", 12))
        self.win_label.grid(row=1, column=0, padx=15, pady=10, sticky="w")
        self.win_entry = ctk.CTkEntry(self.export_frame, width=100, fg_color="#1e293b")
        self.win_entry.insert(0, "16")
        self.win_entry.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        self.win_info = ctk.CTkLabel(self.export_frame, text="(자동 감지됨)", text_color="#94a3b8", font=("Segoe UI", 11))
        self.win_info.grid(row=1, column=2, padx=5, pady=10, sticky="w")

        # Opset Version
        self.opset_label = ctk.CTkLabel(self.export_frame, text="Opset Version:", font=("Segoe UI", 12))
        self.opset_label.grid(row=2, column=0, padx=15, pady=10, sticky="w")
        self.opset_var = ctk.StringVar(value="17")
        self.opset_menu = ctk.CTkOptionMenu(self.export_frame, values=["14", "15", "16", "17", "18"], variable=self.opset_var, width=80, fg_color="#1e293b")
        self.opset_menu.grid(row=2, column=1, padx=5, pady=10, sticky="w")

        # Simplification Switch
        self.sim_var = ctk.BooleanVar(value=True)
        self.sim_switch = ctk.CTkSwitch(self.export_frame, text="ONNX Simplifier 사용", variable=self.sim_var, progress_color="#3b82f6")
        self.sim_switch.grid(row=0, column=3, padx=30, pady=10, sticky="w")

        # Merge Weights Switch
        self.merge_var = ctk.BooleanVar(value=True)
        self.merge_switch = ctk.CTkSwitch(self.export_frame, text="가중치 병합 (Single File)", variable=self.merge_var, progress_color="#10b981")
        self.merge_switch.grid(row=1, column=3, padx=30, pady=10, sticky="w")

        # 변환 버튼
        self.convert_btn = ctk.CTkButton(
            self, 
            text="🚀 ONNX 변환 시작", 
            command=self.start_conversion, 
            height=60, 
            font=("Segoe UI", 20, "bold"), 
            fg_color="#10b981", 
            hover_color="#059669"
        )
        self.convert_btn.pack(pady=20, padx=40, fill="x")

        # 상태 표시바
        self.status_bar = ctk.CTkFrame(self, height=40, fg_color="#0f172a", corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(self.status_bar, text="준비됨", text_color="#94a3b8", font=("Segoe UI", 11))
        self.status_label.pack(pady=5)

    def create_section_label(self, parent, text):
        label = ctk.CTkLabel(parent, text=text, font=("Segoe UI", 15, "bold"), text_color="#60a5fa")
        label.pack(anchor="w", pady=(5, 5))

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="모델 파일 선택",
            filetypes=[("Model Files", "*.safetensors *.pth"), ("Safetensors", "*.safetensors"), ("PyTorch", "*.pth")]
        )
        if filename:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, filename)

    def update_status(self, text, color="#94a3b8"):
        self.status_label.configure(text=text, text_color=color)
        self.update()

    def get_model(self, arch, scale, window_size, img_size):
        if "HAT" in arch:
            if not HAT_AVAILABLE: raise ImportError("hat_arch.py를 찾을 수 없습니다.")
            params = {
                "img_size": img_size, "patch_size": 1, "in_chans": 3, "embed_dim": 180, 
                "depths": (6,6,6,6,6,6), "num_heads": (6,6,6,6,6,6), "window_size": window_size,
                "compress_ratio": 3, "squeeze_factor": 30, "conv_scale": 0.01, 
                "overlap_ratio": 0.5, "mlp_ratio": 2., "upscale": scale, 
                "upsampler": 'pixelshuffle', "resi_connection": '1conv'
            }
            if arch == "HAT-L":
                params["depths"] = (6,6,6,6,6,6,6,6,6,6,6,6)
                params["num_heads"] = (6,6,6,6,6,6,6,6,6,6,6,6)
            elif arch == "HAT-Small":
                params["embed_dim"] = 144
            return HAT(**params)
        
        elif "SwinIR" in arch:
            if not BASI_SR_AVAILABLE: raise ImportError("basicsr이 설치되지 않았습니다.")
            if "Large" in arch:
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240, 
                             num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8], mlp_ratio=2, 
                             upsampler='nearest+conv', resi_connection='3conv')
            elif "Classic" in arch:
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                             num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
            elif "Light" in arch:
                return SwinIR(upscale=scale, in_chans=3, img_size=img_size, window_size=window_size,
                             img_range=1.0, depths=[6, 6, 6, 6], embed_dim=60, 
                             num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

        elif "Real-ESRGAN" in arch:
            if not BASI_SR_AVAILABLE: raise ImportError("basicsr이 설치되지 않았습니다.")
            if "23B" in arch:
                return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            elif "6B" in arch:
                return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
            elif "8B" in arch:
                return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=8, num_grow_ch=32, scale=scale)

        elif "Real-CUGAN" in arch:
            if not CUGAN_AVAILABLE: raise ImportError("cugan_arch.py를 찾을 수 없습니다.")
            return RealCUGAN(in_channels=3, out_channels=3, scale=scale)

        return None

    def start_conversion(self):
        model_path = self.path_entry.get().strip()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("오류", "유효한 모델 파일을 선택해주세요.")
            return

        arch = self.arch_var.get()
        scale = int(self.scale_var.get())
        opset = int(self.opset_var.get())
        
        try:
            h, w = map(int, self.size_entry.get().split(","))
            window_size = int(self.win_entry.get())
        except:
            messagebox.showerror("오류", "입력 크기 또는 윈도우 크기가 올바르지 않습니다.")
            return

        self.convert_btn.configure(state="disabled", text="⏳ 변환 중...")
        self.update_status("가중치 로딩 중...", "#fbbf24")

        try:
            # 1. 가중치 로드
            if model_path.endswith(".safetensors"):
                state_dict = load_safetensors(model_path, device="cpu")
            else:
                state_dict = torch.load(model_path, map_location="cpu")
            
            # Key 추출
            for k in ["params_ema", "params", "state_dict", "model"]:
                if k in state_dict:
                    state_dict = state_dict[k]
                    break
            
            # Module 프리픽스 제거 및 윈도우 사이즈 감지
            new_state_dict = {}
            detected_win = None
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
                if not detected_win and 'relative_position_bias_table' in k and 'overlap_attn' not in k:
                    try:
                        size = int(v.shape[0]**0.5)
                        detected_win = (size + 1) // 2
                    except: pass

            if detected_win and detected_win != window_size:
                window_size = detected_win
                self.win_entry.delete(0, "end")
                self.win_entry.insert(0, str(window_size))
                self.update_status(f"윈도우 크기 감지됨: {window_size}", "#fbbf24")

            # 2. 모델 초기화
            model = self.get_model(arch, scale, window_size, h)
            if model is None: raise ValueError("지원하지 않는 아키텍처입니다.")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(new_state_dict, strict=True)
            model.to(device).eval()

            # 3. ONNX Export
            output_onnx = os.path.splitext(model_path)[0] + ".onnx"
            temp_onnx = output_onnx + ".temp"
            dummy_input = torch.randn(1, 3, h, w).to(device)

            self.update_status("ONNX 내보내기 중...", "#fbbf24")
            torch.onnx.export(
                model, dummy_input, temp_onnx,
                export_params=True, opset_version=opset,
                do_constant_folding=True,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input':{2:'height', 3:'width'}, 'output':{2:'height', 3:'width'}}
            )

            # 4. Simplification
            if self.sim_var.get() and ONNXSIM_AVAILABLE:
                self.update_status("모델 최적화(Simplify) 중...", "#fbbf24")
                try:
                    onnx_model = onnx.load(temp_onnx)
                    model_simp, check = simplify(onnx_model)
                    if check: 
                        onnx.save(model_simp, temp_onnx)
                    else:
                        print("Simplifier check failed.")
                except Exception as e:
                    print(f"Simplifier error: {e}")

            # 5. Weight Merging
            if self.merge_var.get():
                self.update_status("가중치 병합 중...", "#fbbf24")
                try:
                    onnx_obj = onnx.load(temp_onnx, load_external_data=True)
                    onnx.save_model(onnx_obj, output_onnx, save_as_external_data=False)
                    if os.path.exists(temp_onnx): os.remove(temp_onnx)
                    if os.path.exists(temp_onnx + ".data"): os.remove(temp_onnx + ".data")
                except:
                    if os.path.exists(temp_onnx): os.rename(temp_onnx, output_onnx)
            else:
                if os.path.exists(temp_onnx): os.rename(temp_onnx, output_onnx)

            self.update_status(f"완료: {os.path.basename(output_onnx)}", "#34d399")
            messagebox.showinfo("성공", f"성공적으로 변환되었습니다!\n경로: {output_onnx}")

        except Exception as e:
            traceback.print_exc()
            self.update_status("오류 발생", "#f87171")
            messagebox.showerror("변환 오류", f"에러 발생:\n{str(e)}")
        
        finally:
            self.convert_btn.configure(state="normal", text="🚀 ONNX 변환 시작")

if __name__ == "__main__":
    app = SafeToONNXConverter()
    app.mainloop()
