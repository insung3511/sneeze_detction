import torch
import torch.nn as nn
import numpy as np
from realtime_sneeze import YAMNet, SneezeDetector

def diagnose_model():
    """모델 진단 함수"""
    print("=== 모델 진단 시작 ===")
    
    detector = SneezeDetector()
    model = detector.model
    
    if model is None:
        print("모델 로드 실패")
        return
    
    print(f"모델 구조:")
    print(model)
    
    # 다양한 입력 테스트
    test_cases = [
        ("영행렬", np.zeros(32000)),
        ("작은 랜덤", np.random.randn(32000) * 0.01),
        ("중간 랜덤", np.random.randn(32000) * 0.1),
        ("큰 랜덤", np.random.randn(32000) * 1.0),
        ("사인파", np.sin(np.linspace(0, 100 * np.pi, 32000))),
    ]
    
    print("\n=== 다양한 입력 테스트 ===")
    for name, audio_data in test_cases:
        audio_data = audio_data.astype(np.float32)
        is_sneeze, probability = detector.detect_sneeze(audio_data)
        print(f"{name}: 확률 = {probability:.6f}")
    
    # 모델의 마지막 레이어 확인
    print("\n=== 모델의 마지막 레이어 확인 ===")
    print(f"fc 레이어: {model.fc}")
    print(f"sigmoid 레이어: {model.sigmoid}")
    
    # 중간 출력 확인
    print("\n=== 중간 출력 확인 ===")
    dummy_input = torch.randn(1, 1, 32000).to(detector.device)
    
    # 중간 출력을 얻기 위해 forward 함수 수정
    class YAMNetDebug(YAMNet):
        def forward_debug(self, x):
            x = self.conv1(x)
            print(f"conv1 후: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")
            
            x = self.layers(x)
            print(f"layers 후: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")
            
            x = self.gap(x)
            print(f"gap 후: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")
            
            x = x.view(x.size(0), -1)
            print(f"view 후: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")
            
            x = self.fc(x)
            print(f"fc 후: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")
            
            x = self.sigmoid(x)
            print(f"sigmoid 후: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")
            
            return x
    
    debug_model = YAMNetDebug(num_classes=1).to(detector.device)
    debug_model.load_state_dict(torch.load('./models/yamnet_epoch50_val0.57381033.pth', map_location=detector.device), strict=False)
    debug_model.eval()
    
    with torch.no_grad():
        output = debug_model.forward_debug(dummy_input)
        print(f"최종 출력: {output.item():.6f}")

if __name__ == "__main__":
    diagnose_model()