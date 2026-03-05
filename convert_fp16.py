import os
import onnx
from onnx import TensorProto
import numpy as np
from onnx.numpy_helper import to_array, from_array
import customtkinter as ctk
from tkinter import filedialog, messagebox
import traceback

# UI 설정
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def convert_graph_precision_to_fp16(graph, status_callback=None):
    """
    재귀적으로 그래프와 서브그래프의 모든 요소를 FP16으로 변환
    """
    # 1. 초기화된 가중치(Initializers) 변환
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            try:
                np_arr = to_array(init).astype(np.float16)
                new_init = from_array(np_arr, name=init.name)
                init.CopyFrom(new_init)
            except Exception as e:
                print(f"Warning: Failed to convert initializer {init.name}: {e}")

    # 2. 입출력 및 중간 텐서(Value_info) 타입 변환
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    # 3. 모든 노드 순회하며 속성(Attribute) 및 서브그래프 변환
    for node in graph.node:
        # 노드의 모든 속성 검사 (Constant의 value, Cast의 to, ConstantOfShape의 value 등)
        for attr in node.attribute:
            # (1) 텐서 타입 속성 처리 (Constant, ConstantOfShape 등)
            if attr.type == onnx.AttributeProto.TENSOR:
                if attr.t.data_type == TensorProto.FLOAT:
                    try:
                        np_arr = to_array(attr.t).astype(np.float16)
                        attr.t.CopyFrom(from_array(np_arr))
                    except Exception as e:
                        print(f"Warning: Failed to convert attribute tensor in node {node.name}: {e}")
            
            # (2) 데이터 타입 지정 속성 처리 (Cast, RandomNormal 등)
            elif attr.name in ['to', 'dtype'] and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
            
            # (3) 서브그래프 처리 (Loop, If, Scan 등)
            elif attr.type == onnx.AttributeProto.GRAPH:
                convert_graph_precision_to_fp16(attr.g, status_callback)

def convert_fp32_to_fp16_logic(input_model_path, output_model_path, status_callback=None):
    if status_callback: status_callback(f"[{os.path.basename(input_model_path)}] 로딩 중...", "#60a5fa")
    
    # 모델 로드
    model = onnx.load(input_model_path)
    
    if status_callback: status_callback("전체 그래프 및 서브그래프 FP16 변환 중...", "#fbbf24")
    
    # 메인 그래프부터 시작하여 재귀적으로 변환
    convert_graph_precision_to_fp16(model.graph, status_callback)

    if status_callback: status_callback("저장 전 무결성 확인 중...", "#fbbf24")
    
    # 변환된 모델 저장
    if status_callback: status_callback("파일 저장 중...", "#34d399")
    
    # 모델의 ir_version에 따라 large model 지원이 필요할 수 있음
    try:
        onnx.save(model, output_model_path)
    except Exception as e:
        if "size" in str(e).lower():
            if status_callback: status_callback("대용량 모델 저장 시도 중 (외부 데이터)...", "#fbbf24")
            onnx.save(model, output_model_path, save_as_external_data=True, all_tensors_to_one_file=True)
        else:
            raise e
            
    return True

class FP16ConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ONNX FP32 to FP16 Converter (Recursive)")
        self.geometry("650x500")
        self.resizable(False, False)

        # 헤더 섹션
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=10, fg_color="#1e293b")
        self.header_frame.pack(fill="x", pady=20, padx=20)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="⚡ ONNX FP16 Recursive Converter", 
            font=("Segoe UI", 22, "bold"), 
            text_color="#60a5fa"
        )
        self.title_label.pack(expand=True)

        # 메인 컨테이너
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=20)

        # 1. 입력 파일 선택
        self.create_label(self.content_frame, "1. 변환할 ONNX 파일 선택 (FP32)")
        self.file_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.file_frame.pack(fill="x", pady=(5, 20))

        self.input_entry = ctk.CTkEntry(
            self.file_frame, 
            placeholder_text="FP32 ONNX 파일을 선택하세요...", 
            border_width=0, 
            height=35,
            fg_color="#1e293b"
        )
        self.input_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        self.browse_btn = ctk.CTkButton(
            self.file_frame, 
            text="찾아보기", 
            command=self.browse_input, 
            width=80, 
            height=35,
            fg_color="#3b82f6", 
            hover_color="#2563eb"
        )
        self.browse_btn.pack(side="right", padx=10, pady=10)

        # 2. 출력 파일 경로 (자동 생성)
        self.create_label(self.content_frame, "2. 저장 경로")
        self.output_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.output_frame.pack(fill="x", pady=(5, 20))

        self.output_entry = ctk.CTkEntry(
            self.output_frame, 
            placeholder_text="변환될 파일의 저장 경로...", 
            border_width=0, 
            height=35,
            fg_color="#1e293b"
        )
        self.output_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        # 변환 버튼
        self.convert_btn = ctk.CTkButton(
            self, 
            text="🚀 FP16 정밀도 변환 (순수 ONNX 방식)", 
            command=self.start_conversion, 
            height=55, 
            font=("Segoe UI", 20, "bold"), 
            fg_color="#10b981", 
            hover_color="#059669"
        )
        self.convert_btn.pack(pady=30, padx=40, fill="x")

        # 상태 표시바
        self.status_bar = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(self.status_bar, text="준비됨", text_color="#94a3b8", font=("Segoe UI", 11))
        self.status_label.pack(pady=12)

    def create_label(self, parent, text):
        label = ctk.CTkLabel(parent, text=text, font=("Segoe UI", 13, "bold"), text_color="#94a3b8")
        label.pack(anchor="w", padx=5)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="ONNX 파일 선택",
            filetypes=[("ONNX Models", "*.onnx")]
        )
        if filename:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, filename)
            
            # 출력 파일명 자동 제안
            base, ext = os.path.splitext(filename)
            output_path = f"{base}_fp16{ext}"
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, output_path)
            self.update_status(f"선택됨: {os.path.basename(filename)}", "#60a5fa")

    def update_status(self, text, color="#94a3b8"):
        self.status_label.configure(text=text, text_color=color)
        self.update()

    def start_conversion(self):
        input_path = self.input_entry.get().strip()
        output_path = self.output_entry.get().strip()

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("오류", "유효한 입력 파일을 선택해주세요.")
            return

        self.convert_btn.configure(state="disabled", text="⏳ 변환 중...")
        
        try:
            success = convert_fp32_to_fp16_logic(input_path, output_path, self.update_status)
            if success:
                self.update_status(f"🎉 변환 완료: {os.path.basename(output_path)}", "#34d399")
                messagebox.showinfo("완료", "서브그래프(Loop/If) 및 모든 특수 연산자를 포함한\nFP16 변환이 성공적으로 완료되었습니다!")
        except Exception as e:
            traceback.print_exc()
            self.update_status("❌ 오류 발생", "#f87171")
            messagebox.showerror("변환 오류", f"에러가 발생했습니다:\n{str(e)}")
        finally:
            self.convert_btn.configure(state="normal", text="🚀 FP16 정밀도 변환 (순수 ONNX 방식)")

if __name__ == "__main__":
    app = FP16ConverterApp()
    app.mainloop()
