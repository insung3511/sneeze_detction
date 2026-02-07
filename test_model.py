import torch
import torch.nn as nn
import numpy as np

# Let's recreate the exact model architecture from training
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
    def __init__(self, num_classes=1):  # Changed to 1 for binary classification
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

def test_model_loading():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = YAMNet(num_classes=1).to(device)
    
    # Load weights
    model_path = './models/yamnet_epoch50_val0.57381033.pth'
    try:
        saved_weight = torch.load(model_path, map_location=device)
        
        # Try loading with strict=False to see what's missing
        model.load_state_dict(saved_weight, strict=False)
        
        print("Model loaded successfully!")
        
        # Test with dummy data
        dummy_input = torch.randn(1, 1, 32000).to(device)  # 2 seconds at 16kHz
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Model output shape: {output.shape}")
            print(f"Model output value: {output.item():.6f}")
            
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()