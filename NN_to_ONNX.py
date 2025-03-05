import torch
import torch.nn as nn
import torch.onnx

# ✅ 모델을 GPU로 강제 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 간단한 신경망 정의 (예제용, Lang-Seg 모델로 변경 가능)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ✅ 모델과 입력을 GPU로 이동
model = SimpleNN().to(device)
dummy_input = torch.randn(1, 10).to(device)  # 입력도 GPU에 배치

# ✅ 모델을 평가 모드로 설정
model.eval()

# ✅ ONNX 변환 실행 (GPU에서 수행)
onnx_path = "simple_nn.onnx"
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                  opset_version=11)

print(f"✅ ONNX 모델이 GPU에서 변환 완료: {onnx_path}")
