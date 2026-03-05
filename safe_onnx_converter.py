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

# ONNX Simplifier 임포트
try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    simplify = None

import spandrel
from spandrel import ModelLoader

# UI 설정
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SafeToONNXConverter(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Safetensors to ONNX Converter Pro")
        self.geometry("750x800")
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
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
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
        
        self.arch_var = ctk.StringVar(value="Auto-Detect (Spandrel)")
        self.arch_menu = ctk.CTkOptionMenu(
            self.settings_frame, 
            values=["Auto-Detect (Spandrel)"],
            variable=self.arch_var,
            width=250,
            fg_color="#1e293b",
            button_color="#3b82f6",
            state="disabled"
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
        self.opset_var = ctk.StringVar(value="18")
        self.opset_menu = ctk.CTkOptionMenu(self.export_frame, values=["11", "13", "14", "15", "16", "17", "18"], variable=self.opset_var, width=80, fg_color="#1e293b")
        self.opset_menu.grid(row=2, column=1, padx=5, pady=10, sticky="w")

        # Simplification Switch
        self.sim_var = ctk.BooleanVar(value=True)
        self.sim_switch = ctk.CTkSwitch(self.export_frame, text="ONNX Simplifier 사용", variable=self.sim_var, progress_color="#10b981")
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
        self.status_bar = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(self.status_bar, text="준비됨", text_color="#94a3b8", font=("Segoe UI", 11))
        self.status_label.pack(pady=15)

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


    def start_conversion(self):
        model_path = self.path_entry.get().strip()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("오류", "유효한 모델 파일을 선택해주세요.")
            return

        arch_selection = self.arch_var.get()
        scale = int(self.scale_var.get())
        opset = int(self.opset_var.get())
        
        try:
            h, w = map(int, self.size_entry.get().split(","))
        except:
            messagebox.showerror("오류", "더미 입력 크기가 올바르지 않습니다. (예: 256,256)")
            return

        self.convert_btn.configure(state="disabled", text="⏳ 변환 중...")
        self.update_status("모델 로딩 및 분석 중...", "#fbbf24")

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 1. 모델 로드 (Spandrel 우선 사용)
            loader = ModelLoader()
            model = None
            arch_id = "Unknown"
            
            try:
                # Spandrel로 자동 로드 시도
                model_descriptor = loader.load_from_file(model_path)
                model = model_descriptor.model
                arch_id = model_descriptor.architecture.id
                detected_scale = model_descriptor.scale
                self.update_status(f"아키텍처 감지됨: {arch_id} (x{detected_scale})", "#34d399")
                
                if detected_scale != scale:
                    self.scale_var.set(str(detected_scale))
                    scale = detected_scale
            except Exception as spandrel_err:
                self.update_status("Spandrel 자동 감지 실패", "#f87171")
                raise spandrel_err

            model.to(device).eval()

            # 2. 패치 및 설정
            transformer_keywords = ["SwinIR", "Swin2SR", "HAT", "DAT", "DRCT", "Swin"]
            is_transformer = any(k.lower() in arch_id.lower() for k in transformer_keywords) or \
                             any(k.lower() in arch_selection.lower() for k in transformer_keywords)

            # [해결책] 모든 하위 모듈까지 뒤져서 c_mean 버퍼를 일반 속성으로 강제 전환
            def patch_model_recursive(m):
                # _buffers에서 삭제
                if 'c_mean' in m._buffers:
                    val = m._buffers['c_mean'].detach().clone()
                    del m._buffers['c_mean']
                    # 일반 속성으로 할당 (forward에서 self.c_mean으로 접근 가능)
                    setattr(m, 'c_mean', val)
                
                # non_persistent_buffers에서도 삭제
                if hasattr(m, '_non_persistent_buffers_set'):
                    m._non_persistent_buffers_set.discard('c_mean')
                
                # 하위 모듈에 대해서도 동일 작업 수행
                for child in m.children():
                    patch_model_recursive(child)

            self.update_status("모델 아키텍처 호환성 패치 중...", "#fbbf24")
            patch_model_recursive(model)

            # 3. ONNX Export
            output_onnx = os.path.splitext(model_path)[0] + ".onnx"
            temp_onnx = output_onnx + ".temp"
            
            # OOM 및 익스포트 안정성을 위해 CPU 이동
            model.cpu()
            dummy_input = torch.randn(1, 3, h, w).cpu()

            self.update_status("ONNX 파일 생성 중 (안전 모드)...", "#fbbf24")
            
            try:
                with torch.no_grad():
                    torch.onnx.export(
                        model, dummy_input, temp_onnx,
                        export_params=True,
                        opset_version=opset,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        # Full Dynamic Shape 지원 (B, H, W)
                        dynamic_axes={
                            'input': {0: 'batch', 2: 'height', 3: 'width'},
                            'output': {0: 'batch', 2: 'out_height', 3: 'out_width'}
                        }
                    )
            except Exception as e:
                # 만약 위 방법도 실패하면 최후의 수단으로 고정 차원 시도
                err_str = str(e)
                self.update_status(f"재시도 중 (사유: {err_str[:20]}...)", "#fbbf24")
                torch.onnx.export(
                    model, dummy_input, temp_onnx,
                    export_params=True, opset_version=opset,
                    do_constant_folding=True,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch', 2: 'height', 3: 'width'},
                        'output': {0: 'batch', 2: 'out_height', 3: 'out_width'}
                    }
                )

            # 4. Simplification & Dynamic Shape Injections (필수)
            if ONNXSIM_AVAILABLE:
                self.update_status("가변 차원 주입 및 최적화 중...", "#fbbf24")
                try:
                    import onnx
                    from onnxsim import simplify
                    onnx_model = onnx.load(temp_onnx)
                    
                    # 이미 export 단계에서 dynamic_axes가 설정되었으므로 
                    # 추가적인 input_shapes 지정 없이 단순화만 수행합니다.
                    model_simp, check = simplify(onnx_model)
                    
                    if check: 
                        onnx.save(model_simp, temp_onnx)
                        self.update_status("가변 차원 변환 및 최적화 성공", "#34d399")
                    else:
                        self.update_status("최적화 완료 (검증 실패 가능성 있음)", "#fbbf24")
                except Exception as sim_err:
                    print(f"Simplifier error: {sim_err}")
                    self.update_status("가변 주입 실패", "#f87171")

            # 5. Weight Merging
            if self.merge_var.get():
                self.update_status("파일 병합 중...", "#fbbf24")
                try:
                    import onnx
                    onnx_obj = onnx.load(temp_onnx, load_external_data=True)
                    onnx.save_model(onnx_obj, output_onnx, save_as_external_data=False)
                    if os.path.exists(temp_onnx): os.remove(temp_onnx)
                except:
                    if os.path.exists(temp_onnx): os.rename(temp_onnx, output_onnx)
            else:
                if os.path.exists(temp_onnx): os.rename(temp_onnx, output_onnx)

            self.update_status(f"완료: {os.path.basename(output_onnx)}", "#34d399")
            messagebox.showinfo("성공", f"성공적으로 변환되었습니다!\n경로: {output_onnx}")

        except Exception as e:
            traceback.print_exc()
            self.update_status("오류 발생", "#f87171")
            
            error_msg = str(e)
            if "No Adapter To Version $17 for Resize" in error_msg:
                error_msg += "\n\n💡 힌트: Opset 17에서 Resize 연산자 변환 오류가 감지되었습니다. Opset Version을 16 또는 18로 변경하여 다시 시도해보세요."
            elif "topologically sorted" in error_msg or "OpType: Loop" in error_msg:
                error_msg += "\n\n💡 힌트: HAT 모델의 복잡한 구조로 인해 최적화(Simplify) 중 순서 정렬 오류가 발생했습니다.\n\n✅ 해결 방법: Opset Version을 [16]으로 설정하고 다시 시도해보세요. (11이 안 될 경우 16이 가장 안정적입니다.)"
            elif "inferred a static shape" in error_msg or "torch.export" in error_msg:
                error_msg += "\n\n💡 힌트: PyTorch 2.5+ 익스포터가 모델의 특정 연산을 고정 크기로 감지했습니다.\n\n✅ 해결 방법: 가변 차원(Dynamic Shape)이 꼭 필요하다면 PyTorch 2.4.1 버전을 사용하는 것이 가장 권장됩니다. 현재 2.5+라면 [더미 입력 크기]를 실제 사용할 크기로 고정하여 변환하세요."
            elif "No Adapter" in error_msg and "ScatterND" in error_msg:
                error_msg += "\n\n💡 힌트: 최신 PyTorch 익스포터가 낮은 버전(11)으로 변환하지 못하고 있습니다.\n\n✅ 해결 방법: Opset Version을 [16]으로 설정하고 다시 시도해보세요. 16은 최신 기능과 호환성을 모두 갖춘 버전입니다."
            elif "Resize" in error_msg and "opset" in error_msg:
                error_msg += "\n\n💡 힌트: Resize 연산자 관련 Opset 호환성 문제입니다. 다른 Opset 버전을 시도해보세요."
                
            messagebox.showerror("변환 오류", f"에러 발생:\n{error_msg}")
        
        finally:
            self.convert_btn.configure(state="normal", text="🚀 ONNX 변환 시작")

if __name__ == "__main__":
    app = SafeToONNXConverter()
    app.mainloop()
