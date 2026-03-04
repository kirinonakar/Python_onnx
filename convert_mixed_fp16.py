import os
import onnx
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox

# UI 설정
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def convert_to_mixed_fp16(input_path, output_path, keep_ops=['Softmax', 'LayerNormalization'], status_callback=None):
    try:
        if status_callback: status_callback("onnxconverter-common 확인 중...", "#60a5fa")
        from onnxconverter_common import float16
    except ImportError:
        raise ImportError("onnxconverter-common 패키지가 필요합니다.\n'pip install onnxconverter-common' 명령어로 설치해주세요.")

    if status_callback: status_callback(f"[{os.path.basename(input_path)}] 로딩 중...", "#60a5fa")
    model = onnx.load(input_path)

    # 차단 리스트(FP32 유지 대상) 생성
    node_block_list = []
    
    if status_callback: status_callback(f"FP32 유지 노드 찾는 중 ({', '.join(keep_ops)})...", "#fbbf24")
    
    # 그래프 및 모든 서브그래프 순회하며 특정 Op 탐색
    def find_nodes_to_block(graph):
        blocked = []
        for node in graph.node:
            if node.op_type in keep_ops:
                blocked.append(node.name)
            
            # 서브그래프 속성 확인
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    blocked.extend(find_nodes_to_block(attr.g))
        return blocked

    node_block_list = find_nodes_to_block(model.graph)
    
    if status_callback: status_callback(f"발견된 FP32 유지 노드 수: {len(node_block_list)}개", "#fbbf24")

    if status_callback: status_callback("FP16 변환 중 (Mixed Precision)...", "#fbbf24")
    
    # Mixed Precision 변환 실행
    model_fp16 = float16.convert_float_to_float16(
        model,
        node_block_list=node_block_list,
        keep_io_types=False,  # 입출력까지 FP16으로 할지 여부. 필요시 True로 변경 가능
        disable_shape_infer=False
    )

    if status_callback: status_callback("파일 저장 중...", "#34d399")
    onnx.save(model_fp16, output_path)
    return True, len(node_block_list)

class MixedPrecisionConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ONNX Mixed Precision Converter (FP16)")
        self.geometry("700x550")
        self.resizable(False, False)

        # 헤더 섹션
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=10, fg_color="#1e293b")
        self.header_frame.pack(fill="x", pady=20, padx=20)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="⚡ ONNX Mixed Precision Converter", 
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
        self.file_frame.pack(fill="x", pady=(5, 15))

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

        # 2. FP32 유지 옵션
        self.create_label(self.content_frame, "2. FP32 정밀도를 유지할 레이어 (권장)")
        self.ops_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.ops_frame.pack(fill="x", pady=(5, 15))

        self.softmax_var = ctk.BooleanVar(value=True)
        self.softmax_chk = ctk.CTkCheckBox(self.ops_frame, text="Softmax", variable=self.softmax_var, fg_color="#10b981")
        self.softmax_chk.pack(side="left", padx=20, pady=15)

        self.layernorm_var = ctk.BooleanVar(value=True)
        self.layernorm_chk = ctk.CTkCheckBox(self.ops_frame, text="LayerNormalization", variable=self.layernorm_var, fg_color="#10b981")
        self.layernorm_chk.pack(side="left", padx=20, pady=15)

        # 3. 출력 파일 경로
        self.create_label(self.content_frame, "3. 저장 경로")
        self.output_frame = ctk.CTkFrame(self.content_frame, fg_color="#334155")
        self.output_frame.pack(fill="x", pady=(5, 15))

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
            text="🚀 Mixed Precision 변환 시작", 
            command=self.start_conversion, 
            height=55, 
            font=("Segoe UI", 20, "bold"), 
            fg_color="#10b981", 
            hover_color="#059669"
        )
        self.convert_btn.pack(pady=20, padx=40, fill="x")

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
            
            base, ext = os.path.splitext(filename)
            output_path = f"{base}_mixed_fp16{ext}"
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

        keep_ops = []
        if self.softmax_var.get(): keep_ops.append('Softmax')
        if self.layernorm_var.get(): keep_ops.append('LayerNormalization')

        self.convert_btn.configure(state="disabled", text="⏳ 변환 중...")
        
        try:
            success, count = convert_to_mixed_fp16(input_path, output_path, keep_ops, self.update_status)
            if success:
                self.update_status(f"🎉 변환 완료: {os.path.basename(output_path)}", "#34d399")
                messagebox.showinfo("완료", f"성공적으로 변환되었습니다!\n\nFP32 유지 노드: {count}개\n나머지: FP16 변환됨")
        except ImportError as ie:
            self.update_status("❌ 패키지 누락", "#f87171")
            messagebox.showerror("라이브러리 필요", str(ie))
        except Exception as e:
            traceback.print_exc()
            self.update_status("❌ 오류 발생", "#f87171")
            messagebox.showerror("변환 오류", f"에러가 발생했습니다:\n{str(e)}")
        finally:
            self.convert_btn.configure(state="normal", text="🚀 Mixed Precision 변환 시작")

if __name__ == "__main__":
    app = MixedPrecisionConverterApp()
    app.mainloop()
