import onnx
from onnx import TensorProto
import numpy as np
from onnx.numpy_helper import to_array, from_array

def convert_int64_to_int32(input_model_path, output_model_path):
    print(f"[{input_model_path}] 모델 로딩 중...")
    model = onnx.load(input_model_path)
    graph = model.graph

    # 1. 초기화된 가중치 및 상수(Initializer) 변환
    print("가중치 및 상수(int64)를 int32로 변환 중...")
    for init in graph.initializer:
        if init.data_type == TensorProto.INT64:
            # 64비트 배열을 32비트로 강제 다운캐스팅
            np_arr = to_array(init).astype(np.int32)
            new_init = from_array(np_arr, name=init.name)
            init.CopyFrom(new_init)

    # 2. 모델의 입력, 출력, 중간 텐서(Value_info) 타입 변환
    print("텐서 입출력 속성을 int32로 변환 중...")
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.type.tensor_type.elem_type == TensorProto.INT64:
            value_info.type.tensor_type.elem_type = TensorProto.INT32

    # 3. Cast 및 Constant 노드 속성 강제 변환
    # (모델 내부에서 실행 도중 int64로 변환하려는 시도를 원천 차단)
    print("내부 노드 연산(Cast, Constant) 강제 변환 중...")
    for node in graph.node:
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == TensorProto.INT64:
                    attr.i = TensorProto.INT32  # int64로 캐스팅하려던 것을 int32로 변경
                    
        elif node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value' and attr.t.data_type == TensorProto.INT64:
                    np_arr = to_array(attr.t).astype(np.int32)
                    attr.t.CopyFrom(from_array(np_arr))

    # 4. 변환된 모델 저장
    onnx.save(model, output_model_path)
    print(f"🎉 변환 완료! 저장된 파일: {output_model_path}")

# ==========================================
# 실행 부분: 파일명을 본인 환경에 맞게 수정하세요.
# ==========================================
input_file = "Real_HAT_GAN_SRx4.onnx"       # 아까 simplifier로 단순화한 파일
output_file = "Real_HAT_GAN_SRx4_256_sim_int32.onnx" # 최종적으로 Rust에서 사용할 파일

convert_int64_to_int32(input_file, output_file)