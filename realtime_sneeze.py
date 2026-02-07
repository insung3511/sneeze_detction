
import threading

import numpy as np

import torch.nn as nn
import torch

import datetime
import pyaudio
import wave
import os

from collections import deque

THRESHOLD = 0.5
LOAD_MODEL_PATH = './models/yamnet_epoch5_val0.58147465.pth'

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
    def __init__(self, num_classes=521):
        super(YAMNet, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(1, 32, kernel_size=3, stride=1, padding=1)
        
        # Depthwise separable convolutions with different configurations
        self.layers = nn.Sequential(
            # Block 1
            DepthwiseConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
            
            # Block 2
            DepthwiseConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            
            # Block 3
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            
            # Block 4
            DepthwiseConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(256, 256, kernel_size=3, stride=2, padding=1),
            
            # Block 5
            DepthwiseConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(256, 256, kernel_size=3, stride=2, padding=1),
            
            # Block 6
            DepthwiseConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            DepthwiseConvBlock(512, 512, kernel_size=3, stride=2, padding=1),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class SneezeDetector:
    def __init__(self, model_path='./models/yamnet_epoch5_val0.58147465.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.sample_rate = 16000
        self.chunk_duration = 2  # 2초
        self.chunk_samples = self.sample_rate * self.chunk_duration
        
        # 오디오 버퍼
        self.audio_buffer = deque(maxlen=int(self.chunk_samples * 2))  # 4초 버퍼
        
        # PyAudio 설정
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 저장 디렉토리
        self.save_dir = 'detected_sneezes'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 모델 로드
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """학습된 모델 로드"""
        try:
            model = YAMNet(num_classes=1).to(self.device)
            saved_weight = torch.load(model_path, map_location=self.device)
            model.load_state_dict(saved_weight, strict=False)
            model.eval()
            print(f"모델 로드 성공")
            return model
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 입력 콜백 함수"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)
    
    def preprocess_audio(self, audio_chunk):
        """오디오 데이터 전처리"""
        # 텐서로 변환 (batch_size=1, channels=1, length=audio_length)
        audio_tensor = torch.FloatTensor(audio_chunk).unsqueeze(0).unsqueeze(0)
        
        # 정규화
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            audio_tensor = audio_tensor / max_val
        
        return audio_tensor.to(self.device)
    
    def detect_sneeze(self, audio_chunk):
        """재채기 감지"""
        if self.model is None:
            return False, 0.0
        
        try:
            # 오디오 전처리
            audio_tensor = self.preprocess_audio(audio_chunk)
            
            # 모델 예측
            with torch.no_grad():
                prediction = self.model(audio_tensor)
                # 이미 sigmoid가 모델 내부에서 적용됨
                probability = prediction.item()
                
            # 재채기 판단
            is_sneeze = probability > THRESHOLD
            
            return is_sneeze, probability
            
        except Exception as e:
            print(f"재채기 감지 중 오류: {e}")
            return False, 0.0
    
    def save_detected_audio(self, audio_chunk, probability):
        """감지된 오디오 저장"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_detected_sneeze.wav"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            # WAV 파일로 저장
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)  # 모노
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                
                # float32를 int16으로 변환
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"🤧 재채기 감지! 저장됨: {filename} (확률: {probability:.3f})")
            
        except Exception as e:
            print(f"오디오 저장 중 오류: {e}")
    
    def should_detect(self, probability):
        """추가적인 감지 로직"""
        # 너무 낮은 확률은 무시
        if probability < 0.3:
            return False
        
        # 확률이 0.9 이상이면 매우 강한 신호로 간주
        if probability >= 0.9:
            return True
        
        # 중간 확률은 임계값으로 판단
        return probability > THRESHOLD
    
    def start_detection(self):
        """실시간 재채기 감지 시작"""
        print("실시간 재채기 감지 시스템 시작...")
        print(f"사용 디바이스: {self.device}")
        print(f"모델 상태: {'로드됨' if self.model is not None else '로드 실패'}")
        print(f"임계값: {THRESHOLD}")
        
        if self.model is None:
            print("모델이 로드되지 않았습니다. 프로그램을 종료합니다.")
            return
        
        try:
            # 오디오 스트림 설정
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            self.stream.start_stream()
            print("마이크 활성화 완료. 재채기 감지 중...")
            print("실시간 감지 시작 (종료하려면 Ctrl+C)")
            
            detection_count = 0
            
            # 메인 감지 루프
            while True:
                if len(self.audio_buffer) >= self.chunk_samples:
                    # 2초 분량의 오디오 데이터 추출
                    audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_samples:])
                    
                    # 재채기 감지
                    is_sneeze, probability = self.detect_sneeze(audio_chunk)
                    
                    # 디버그 출력 (매 10초마다 또는 높은 확률일 때)
                    detection_count += 1
                    if detection_count % 100 == 0 or probability > 0.3:
                        print(f"감지 #{detection_count}: 확률 = {probability:.4f}, 버퍼 = {len(self.audio_buffer)}")
                    
                    # 추가 감지 로직 적용
                    if is_sneeze and self.should_detect(probability):
                        self.save_detected_audio(audio_chunk, probability)
                    elif probability > 0.3:  # 잠재적 재채기 신호
                        print(f"  ⚠️  잠재적 재채기 신호: {probability:.3f}")
                
                # CPU 부하를 줄이기 위한 대기
                threading.Event().wait(0.1)
                
        except KeyboardInterrupt:
            print("\n감지 시스템 중지...")
        except Exception as e:
            print(f"감지 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_detection()
    
    def stop_detection(self):
        """감지 시스템 중지"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("시스템 중지 완료")

def test_model():
    """모델 테스트 함수"""
    print("모델 테스트를 시작합니다...")
    detector = SneezeDetector()
    
    if detector.model is None:
        print("모델 로드 실패 - 테스트를 중단합니다.")
        return
    
    print(f"모델 로드 성공! 디바이스: {detector.device}")
    
    # 더미 오디오 데이터 생성 (2초 분량)
    dummy_audio = np.random.randn(detector.chunk_samples).astype(np.float32)
    print(f"테스트 오디오 길이: {len(dummy_audio)} 샘플 ({detector.chunk_duration}초)")
    
    try:
        is_sneeze, probability = detector.detect_sneeze(dummy_audio)
        print(f"테스트 결과: 재채기 = {is_sneeze}, 확률 = {probability:.6f}")
        print("모델 테스트 성공!")
        
        # 여러 번 테스트하여 출력 범위 확인
        print("\n추가 테스트 (5번):")
        for i in range(5):
            dummy_audio = np.random.randn(detector.chunk_samples).astype(np.float32)
            is_sneeze, probability = detector.detect_sneeze(dummy_audio)
            print(f"  테스트 {i+1}: 확률 = {probability:.6f}")
            
    except Exception as e:
        print(f"모델 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 먼저 모델 테스트
    test_model()
    
    try:
        print("\n실시간 감지를 시작하려면 Enter를 누르세요...")
        input()
        
        detector = SneezeDetector()
        detector.start_detection()
    except EOFError:
        print("\n실시간 감지를 바로 시작합니다...")
        detector = SneezeDetector()
        detector.start_detection()

if __name__ == "__main__":
    main()