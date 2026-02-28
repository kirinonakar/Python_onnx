import torch
import onnx
from ben2 import BEN_Base 

print("1. BEN2 모델 불러오는 중...")
device = torch.device('cpu')
model = BEN_Base().to(device).eval()
model.loadcheckpoints("./BEN2_Base.pth")

# .half()를 빼고 기본 32비트(f32) 텐서를 사용합니다.
dummy_input = torch.randn(1, 3, 1024, 1024) 

print("2. ONNX 임시 파일 생성 중...")
torch.onnx.export(
    model, 
    dummy_input, 
    "ben2_temp.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("3. 단일 파일로 병합 중...")
# 임시로 만들어진 분할 파일들을 하나로 꽉 뭉쳐줍니다.
onnx_model = onnx.load("ben2_temp.onnx", load_external_data=True)
onnx.save_model(onnx_model, "ben2_final.onnx", save_as_external_data=False)

print("완료! 'ben2_final.onnx' 파일 하나만 챙기시면 됩니다!")