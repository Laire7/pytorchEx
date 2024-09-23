import torch

# CUDA 사용 가능한지 확인
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 장치 정보 출력
print(f'Using device: {device}')

# 예시로 텐서로 생성하고 장치로 이동
x = torch.rand(3,3)
x = x.to(device)