import threading
import numpy as np
import torch.nn as nn
import torch
import datetime
import pyaudio
import wave
import os
from collections import deque

THRESHOLD = 0.8
LOAD_MODEL_PATH = 'fine_tuned_model/yamnet_finetuned_epoch10_val0.58175173.pth'

# YAMNet PyTorch Implementation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, 
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class YAMNet(nn.Module):
    def __init__(self, num_classes=1):
        super(YAMNet, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(1, 32, kernel_size=3, stride=1, padding=1)
        
        # Depthwise separable convolutions
        self.layers = nn.Sequential(
            DepthwiseConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(256, 256, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(256, 256, kernel_size=3, stride=2, padding=1),
            DepthwiseConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(512, 512, kernel_size=3, stride=2, padding=1),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class SneezeDetector:
    def __init__(self, model_path=LOAD_MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.sample_rate = 16000
        self.chunk_duration = 2
        self.chunk_samples = self.sample_rate * self.chunk_duration
        
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=int(self.chunk_samples * 2))
        
        # PyAudio 설정
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 저장 디렉토리
        self.save_dir = 'detected_sneezes'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 모델 로드
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            model = YAMNet(num_classes=1).to(self.device)
            saved_weight = torch.load(model_path, map_location=self.device)
            model.load_state_dict(saved_weight, strict=False)
            model.eval()
            print(f"✅ 모델 로드 성공: {model_path}")
            return model
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)
    
    def preprocess_audio(self, audio_chunk):
        # 텐서로 변환
        audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0).unsqueeze(0)
        
        # 0 to 1 정규화
        min_val = torch.min(audio_tensor)
        max_val = torch.max(audio_tensor)
        if max_val > min_val:
            audio_tensor = (audio_tensor - min_val) / (max_val - min_val)
        else:
            audio_tensor = torch.zeros_like(audio_tensor)
        
        return audio_tensor.to(self.device)
    
    def detect_sneeze(self, audio_chunk):
        if self.model is None:
            return False, 0.0
        
        try:
            audio_tensor = self.preprocess_audio(audio_chunk)
            
            with torch.no_grad():
                prediction = self.model(audio_tensor)
                probability = prediction.item()
            
            is_sneeze = probability > THRESHOLD
            return is_sneeze, probability
            
        except Exception as e:
            print(f"❌ 재채기 감지 중 오류: {e}")
            return False, 0.0
    
    def save_detected_audio(self, audio_chunk, probability):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_detected_sneeze.wav"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"🤧 재채기 감지! 저장됨: {filename} (확률: {probability:.3f})")
            
        except Exception as e:
            print(f"❌ 오디오 저장 중 오류: {e}")
    
    def start_detection(self):
        print("🎤 실시간 재채기 감지 시스템 시작...")
        print(f"📱 사용 디바이스: {self.device}")
        print(f"🧠 모델: {self.model_path}")
        print(f"🎯 임계값: {THRESHOLD}")
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다. 프로그램을 종료합니다.")
            return
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            print("🎙️  마이크 활성화 완료. 재채기 감지 중...")
            print("⏹️  종료하려면 Ctrl+C를 누르세요")
            
            detection_count = 0
            
            while True:
                if len(self.audio_buffer) >= self.chunk_samples:
                    audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_samples:])
                    is_sneeze, probability = self.detect_sneeze(audio_chunk)
                    
                    detection_count += 1
                    
                    # 디버그 출력
                    if detection_count % 100 == 0 or probability > 0.3:
                        print(f"🔍 감지 #{detection_count}: 확률 = {probability:.4f}")
                    
                    if is_sneeze:
                        self.save_detected_audio(audio_chunk, probability)
                
                threading.Event().wait(0.1)
                
        except KeyboardInterrupt:
            print("\n👋 감지 시스템 중지...")
        except Exception as e:
            print(f"❌ 감지 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("✅ 시스템 중지 완료")

def test_model():
    print("🧪 모델 테스트 시작...")
    detector = SneezeDetector()
    
    if detector.model is None:
        print("❌ 모델 로드 실패")
        return
    
    print(f"✅ 모델 로드 성공! 디바이스: {detector.device}")
    
    # 다양한 테스트 케이스
    test_cases = [
        ("영행렬", np.zeros(detector.chunk_samples)),
        ("랜덤 노이즈", np.random.randn(detector.chunk_samples).astype(np.float32) * 0.1),
    ]
    
    for name, audio_data in test_cases:
        is_sneeze, probability = detector.detect_sneeze(audio_data)
        print(f"📊 {name}: 재채기={is_sneeze}, 확률={probability:.4f}")
    
    print("✅ 모델 테스트 완료!")

def main():
    test_model()
    
    print("\n🚀 실시간 감지를 시작하려면 Enter를 누르세요...")
    try:
        input()
    except EOFError:
        print("🚀 바로 시작합니다...")
    
    detector = SneezeDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()