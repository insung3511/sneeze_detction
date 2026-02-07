import json

# 셀 생성 헬퍼 함수
def create_code_cell(source, cell_id=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or "",
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

def create_markdown_cell(source, cell_id=None):
    return {
        "cell_type": "markdown",
        "id": cell_id or "",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

# 노트북 셀 정의
cells = []

# Header
cells.append(create_markdown_cell([
    "# 🎯 재채기 탐지 경량화 모델\n",
    "\n",
    "## 목표\n",
    "- Raspberry Pi / Jetson Nano 등 임베디드 시스템에서 실시간 처리 가능한 경량 모델\n",
    "- 개선된 Negative 샘플 (생활 소음)\n",
    "- 데이터 증강으로 성능 향상\n",
    "- MFCC 기반 효율적 특징 추출\n",
    "\n",
    "## 파이프라인\n",
    "1. 데이터 재구성 (생활 소음 중심 Negative 샘플)\n",
    "2. 데이터 증강\n",
    "3. PyTorch Dataset/DataLoader\n",
    "4. 경량 CNN 모델\n",
    "5. 학습 및 평가"
]))

# 1. 임포트
cells.append(create_markdown_cell(["## 1. 라이브러리 임포트"]))
cells.append(create_code_cell([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import librosa\n",
    "import librosa.display\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "import torchaudio\n",
    "import polars as pl\n",
    "\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from tqdm import tqdm\n",
    "import os\n",
    "import random\n",
    "import json\n",
    "\n",
    "# 시드 고정\n",
    "SEED = 42\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "torch.manual_seed(SEED)\n",
    "\n",
    "# 디바이스 설정\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "# 경로 설정\n",
    "ESC50_AUDIO_PATH = \"./esc-50/audio/\"\n",
    "ESC50_META_PATH = \"./esc-50/meta/esc50.csv\"\n",
    "SNEEZE_AUDIO_PATH = \"./datasets/\"\n",
    "MODEL_SAVE_PATH = \"./models/\"\n",
    "os.makedirs(MODEL_SAVE_PATH, exist_ok=True)"
]))

# 2. ESC-50 카테고리 분석
cells.append(create_markdown_cell(["## 2. ESC-50 데이터셋 분석 및 카테고리 선택"]))
cells.append(create_code_cell([
    "# ESC-50 메타데이터 로드\n",
    "desc_csv = pl.read_csv(ESC50_META_PATH)\n",
    "\n",
    "# 생활 소음 관련 카테고리 정의\n",
    "LIFE_NOISE_CATEGORIES = [\n",
    "    'coughing', 'breathing', 'laughing', 'crying_baby',\n",
    "    'footsteps', 'door_wood_knock', 'clapping',\n",
    "    'keyboard_typing', 'drinking_sipping', 'brushing_teeth',\n",
    "]\n",
    "\n",
    "# 존재하는 카테고리만 필터링\n",
    "all_categories = desc_csv['category'].unique().to_list()\n",
    "LIFE_NOISE_CATEGORIES = [cat for cat in LIFE_NOISE_CATEGORIES if cat in all_categories]\n",
    "\n",
    "print(f\"선택된 생활 소음 카테고리 ({len(LIFE_NOISE_CATEGORIES)}개):\")\n",
    "for cat in LIFE_NOISE_CATEGORIES:\n",
    "    count = len(desc_csv.filter(pl.col('category') == cat))\n",
    "    print(f\"  - {cat}: {count} samples\")"
]))

# 3. 데이터 로딩 함수
cells.append(create_markdown_cell(["## 3. 데이터 로딩 함수"]))
cells.append(create_code_cell([
    "def load_audio(file_path, target_sr=16000, target_length=32000):\n",
    "    audio, sr = torchaudio.load(file_path)\n",
    "    if sr != target_sr:\n",
    "        resampler = torchaudio.transforms.Resample(sr, target_sr)\n",
    "        audio = resampler(audio)\n",
    "    if audio.shape[0] > 1:\n",
    "        audio = torch.mean(audio, dim=0, keepdim=True)\n",
    "    if audio.shape[1] > target_length:\n",
    "        audio = audio[:, :target_length]\n",
    "    elif audio.shape[1] < target_length:\n",
    "        audio = F.pad(audio, (0, target_length - audio.shape[1]))\n",
    "    return audio.numpy().flatten()\n",
    "\n",
    "def load_sneeze_dataset(audio_path):\n",
    "    files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]\n",
    "    dataset = []\n",
    "    for f in tqdm(files, desc=\"Sneeze\"):\n",
    "        dataset.append(load_audio(os.path.join(audio_path, f)))\n",
    "    return dataset\n",
    "\n",
    "def load_life_noise_dataset(esc_path, meta_path, categories, max_samples=None):\n",
    "    csv = pl.read_csv(meta_path)\n",
    "    filtered = csv.filter(pl.col('category').is_in(categories))\n",
    "    if max_samples and len(filtered) > max_samples:\n",
    "        filtered = filtered.sample(n=max_samples, seed=SEED)\n",
    "    dataset = []\n",
    "    for row in tqdm(filtered.iter_rows(), total=len(filtered), desc=\"Life Noise\"):\n",
    "        dataset.append(load_audio(os.path.join(esc_path, row[0])))\n",
    "    return dataset\n",
    "\n",
    "# 데이터 로드\n",
    "print(\"=\"*70)\n",
    "print(\"데이터 로딩\")\n",
    "print(\"=\"*70)\n",
    "sneeze_samples = load_sneeze_dataset(SNEEZE_AUDIO_PATH)\n",
    "life_noise_samples = load_life_noise_dataset(\n",
    "    ESC50_AUDIO_PATH, ESC50_META_PATH, LIFE_NOISE_CATEGORIES,\n",
    "    max_samples=len(sneeze_samples)\n",
    ")\n",
    "print(f\"Sneeze: {len(sneeze_samples)}, Life Noise: {len(life_noise_samples)}\")"
]))

# 4. 데이터 증강
cells.append(create_markdown_cell(["## 4. 데이터 증강"]))
cells.append(create_code_cell([
    "def time_stretch(audio, rate_range=(0.8, 1.2)):\n",
    "    rate = np.random.uniform(*rate_range)\n",
    "    stretched = librosa.effects.time_stretch(audio, rate=rate)\n",
    "    if len(stretched) > len(audio):\n",
    "        return stretched[:len(audio)]\n",
    "    else:\n",
    "        return np.pad(stretched, (0, len(audio) - len(stretched)))\n",
    "\n",
    "def pitch_shift(audio, sr=16000, n_steps_range=(-3, 3)):\n",
    "    n_steps = np.random.uniform(*n_steps_range)\n",
    "    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)\n",
    "\n",
    "def add_noise(audio, noise_factor_range=(0.002, 0.01)):\n",
    "    factor = np.random.uniform(*noise_factor_range)\n",
    "    return audio + factor * np.random.randn(len(audio))\n",
    "\n",
    "def time_shift(audio, shift_range=(-0.2, 0.2)):\n",
    "    shift = int(np.random.uniform(*shift_range) * len(audio))\n",
    "    return np.roll(audio, shift)\n",
    "\n",
    "def augment_audio(audio):\n",
    "    aug_type = np.random.choice(['time_stretch', 'pitch_shift', 'add_noise', 'time_shift'])\n",
    "    if aug_type == 'time_stretch':\n",
    "        return time_stretch(audio)\n",
    "    elif aug_type == 'pitch_shift':\n",
    "        return pitch_shift(audio)\n",
    "    elif aug_type == 'add_noise':\n",
    "        return add_noise(audio)\n",
    "    else:\n",
    "        return time_shift(audio)\n",
    "\n",
    "# 데이터 증강 적용\n",
    "print(\"데이터 증강 중...\")\n",
    "sneeze_aug = [augment_audio(a) for a in tqdm(sneeze_samples, desc=\"Sneeze Aug\")]\n",
    "life_aug = [augment_audio(a) for a in tqdm(life_noise_samples, desc=\"Life Aug\")]\n",
    "\n",
    "all_sneeze = sneeze_samples + sneeze_aug\n",
    "all_life_noise = life_noise_samples + life_aug\n",
    "print(f\"증강 후: Sneeze {len(all_sneeze)}, Life Noise {len(all_life_noise)}\")"
]))

# 5. MFCC 추출
cells.append(create_markdown_cell(["## 5. MFCC 특성 추출"]))
cells.append(create_code_cell([
    "def preprocess_audio(audio):\n",
    "    rms = np.sqrt(np.mean(audio**2))\n",
    "    if rms > 0:\n",
    "        audio = audio / rms * 0.1\n",
    "    pre_emphasis = 0.97\n",
    "    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])\n",
    "    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)\n",
    "    if len(audio_trimmed) < len(audio):\n",
    "        audio_trimmed = np.pad(audio_trimmed, (0, len(audio) - len(audio_trimmed)))\n",
    "    elif len(audio_trimmed) > len(audio):\n",
    "        audio_trimmed = audio_trimmed[:len(audio)]\n",
    "    return audio_trimmed\n",
    "\n",
    "def extract_mfcc_features(audio, n_mfcc=20, include_deltas=True):\n",
    "    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)\n",
    "    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)\n",
    "    if include_deltas:\n",
    "        delta = librosa.feature.delta(mfcc)\n",
    "        delta2 = librosa.feature.delta(mfcc, order=2)\n",
    "        mfcc = np.vstack([mfcc, delta, delta2])\n",
    "    return mfcc\n",
    "\n",
    "# MFCC 추출\n",
    "print(\"MFCC 추출 중...\")\n",
    "mfcc_sneeze = []\n",
    "for a in tqdm(all_sneeze, desc=\"Sneeze MFCC\"):\n",
    "    mfcc_sneeze.append(extract_mfcc_features(preprocess_audio(a)))\n",
    "\n",
    "mfcc_life = []\n",
    "for a in tqdm(all_life_noise, desc=\"Life MFCC\"):\n",
    "    mfcc_life.append(extract_mfcc_features(preprocess_audio(a)))\n",
    "\n",
    "print(f\"MFCC shape: {mfcc_sneeze[0].shape}\")"
]))

# 6. 시각화
cells.append(create_markdown_cell(["## 6. 데이터 시각화"]))
cells.append(create_code_cell([
    "fig, axes = plt.subplots(2, 4, figsize=(20, 8))\n",
    "for i in range(4):\n",
    "    axes[0, i].imshow(mfcc_sneeze[i], aspect='auto', origin='lower', cmap='viridis')\n",
    "    axes[0, i].set_title(f'Sneeze #{i+1}', color='red')\n",
    "    axes[1, i].imshow(mfcc_life[i], aspect='auto', origin='lower', cmap='viridis')\n",
    "    axes[1, i].set_title(f'Life Noise #{i+1}', color='blue')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# 7. Dataset/DataLoader
cells.append(create_markdown_cell(["## 7. PyTorch Dataset 및 DataLoader"]))
cells.append(create_code_cell([
    "class SneezeDataset(Dataset):\n",
    "    def __init__(self, mfcc_list, labels):\n",
    "        self.mfcc_list = mfcc_list\n",
    "        self.labels = labels\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.mfcc_list)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        mfcc = torch.FloatTensor(self.mfcc_list[idx]).unsqueeze(0)\n",
    "        label = torch.LongTensor([self.labels[idx]])\n",
    "        return mfcc, label\n",
    "\n",
    "# 데이터셋 준비\n",
    "all_mfcc = mfcc_sneeze + mfcc_life\n",
    "all_labels = [1]*len(mfcc_sneeze) + [0]*len(mfcc_life)\n",
    "\n",
    "# Train/Val/Test 분할\n",
    "train_mfcc, temp_mfcc, train_labels, temp_labels = train_test_split(\n",
    "    all_mfcc, all_labels, test_size=0.3, random_state=SEED, stratify=all_labels\n",
    ")\n",
    "val_mfcc, test_mfcc, val_labels, test_labels = train_test_split(\n",
    "    temp_mfcc, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels\n",
    ")\n",
    "\n",
    "# DataLoader\n",
    "BATCH_SIZE = 32\n",
    "train_loader = DataLoader(SneezeDataset(train_mfcc, train_labels), batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader = DataLoader(SneezeDataset(val_mfcc, val_labels), batch_size=BATCH_SIZE)\n",
    "test_loader = DataLoader(SneezeDataset(test_mfcc, test_labels), batch_size=BATCH_SIZE)\n",
    "\n",
    "print(f\"Train: {len(train_mfcc)}, Val: {len(val_mfcc)}, Test: {len(test_mfcc)}\")"
]))

# 8. 모델 정의
cells.append(create_markdown_cell(["## 8. 경량 CNN 모델"]))
cells.append(create_code_cell([
    "class LightweightSneezeCNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        # Depthwise Separable Conv Blocks\n",
    "        self.conv1_dw = nn.Conv2d(1, 1, 3, padding=1, groups=1)\n",
    "        self.conv1_pw = nn.Conv2d(1, 32, 1)\n",
    "        self.bn1 = nn.BatchNorm2d(32)\n",
    "        self.pool1 = nn.MaxPool2d(2)\n",
    "        \n",
    "        self.conv2_dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)\n",
    "        self.conv2_pw = nn.Conv2d(32, 64, 1)\n",
    "        self.bn2 = nn.BatchNorm2d(64)\n",
    "        self.pool2 = nn.MaxPool2d(2)\n",
    "        \n",
    "        self.conv3_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64)\n",
    "        self.conv3_pw = nn.Conv2d(64, 128, 1)\n",
    "        self.bn3 = nn.BatchNorm2d(128)\n",
    "        self.pool3 = nn.MaxPool2d(2)\n",
    "        \n",
    "        self.gap = nn.AdaptiveAvgPool2d(1)\n",
    "        self.fc1 = nn.Linear(128, 64)\n",
    "        self.dropout = nn.Dropout(0.3)\n",
    "        self.fc2 = nn.Linear(64, 2)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = F.relu(self.bn1(self.conv1_pw(self.conv1_dw(x))))\n",
    "        x = self.pool1(x)\n",
    "        x = F.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))\n",
    "        x = self.pool2(x)\n",
    "        x = F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))\n",
    "        x = self.pool3(x)\n",
    "        x = self.gap(x).view(x.size(0), -1)\n",
    "        x = F.relu(self.fc1(x))\n",
    "        x = self.dropout(x)\n",
    "        return self.fc2(x)\n",
    "\n",
    "model = LightweightSneezeCNN().to(device)\n",
    "total_params = sum(p.numel() for p in model.parameters())\n",
    "print(f\"모델 파라미터: {total_params:,}개 (~{total_params*4/1024/1024:.2f} MB)\")"
]))

# 9. 학습 설정
cells.append(create_markdown_cell(["## 9. 학습 설정"]))
cells.append(create_code_cell([
    "NUM_EPOCHS = 50\n",
    "LEARNING_RATE = 0.001\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)\n",
    "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)\n",
    "\n",
    "print(f\"학습 준비 완료: {NUM_EPOCHS} epochs, LR={LEARNING_RATE}\")"
]))

# 10. 학습 루프
cells.append(create_markdown_cell(["## 10. 학습 루프"]))
cells.append(create_code_cell([
    "def train_epoch(model, loader, criterion, optimizer, device):\n",
    "    model.train()\n",
    "    total_loss, correct, total = 0, 0, 0\n",
    "    for inputs, labels in tqdm(loader, desc=\"Train\"):\n",
    "        inputs, labels = inputs.to(device), labels.squeeze().to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(inputs)\n",
    "        loss = criterion(outputs, labels)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item() * inputs.size(0)\n",
    "        _, pred = torch.max(outputs, 1)\n",
    "        correct += (pred == labels).sum().item()\n",
    "        total += labels.size(0)\n",
    "    return total_loss / total, correct / total\n",
    "\n",
    "def validate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    total_loss, correct, total = 0, 0, 0\n",
    "    all_preds, all_labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in tqdm(loader, desc=\"Val\"):\n",
    "            inputs, labels = inputs.to(device), labels.squeeze().to(device)\n",
    "            outputs = model(inputs)\n",
    "            loss = criterion(outputs, labels)\n",
    "            total_loss += loss.item() * inputs.size(0)\n",
    "            _, pred = torch.max(outputs, 1)\n",
    "            correct += (pred == labels).sum().item()\n",
    "            total += labels.size(0)\n",
    "            all_preds.extend(pred.cpu().numpy())\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "    return total_loss / total, correct / total, all_preds, all_labels\n",
    "\n",
    "# 학습\n",
    "history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}\n",
    "best_val_acc = 0\n",
    "\n",
    "for epoch in range(NUM_EPOCHS):\n",
    "    print(f\"\\nEpoch {epoch+1}/{NUM_EPOCHS}\")\n",
    "    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)\n",
    "    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)\n",
    "    \n",
    "    history['train_loss'].append(train_loss)\n",
    "    history['train_acc'].append(train_acc)\n",
    "    history['val_loss'].append(val_loss)\n",
    "    history['val_acc'].append(val_acc)\n",
    "    \n",
    "    scheduler.step(val_loss)\n",
    "    \n",
    "    print(f\"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%\")\n",
    "    print(f\"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%\")\n",
    "    \n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))\n",
    "        print(f\"✓ Best model saved! (Acc: {val_acc*100:.2f}%)\")\n",
    "\n",
    "print(f\"\\n학습 완료! Best Val Acc: {best_val_acc*100:.2f}%\")"
]))

# 11. 결과 시각화
cells.append(create_markdown_cell(["## 11. 학습 결과 시각화"]))
cells.append(create_code_cell([
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
    "\n",
    "axes[0].plot(history['train_loss'], 'o-', label='Train Loss')\n",
    "axes[0].plot(history['val_loss'], 's-', label='Val Loss')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].set_title('Loss')\n",
    "axes[0].legend()\n",
    "axes[0].grid(True)\n",
    "\n",
    "axes[1].plot([a*100 for a in history['train_acc']], 'o-', label='Train Acc')\n",
    "axes[1].plot([a*100 for a in history['val_acc']], 's-', label='Val Acc')\n",
    "axes[1].set_xlabel('Epoch')\n",
    "axes[1].set_ylabel('Accuracy (%)')\n",
    "axes[1].set_title('Accuracy')\n",
    "axes[1].legend()\n",
    "axes[1].grid(True)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# 12. 테스트
cells.append(create_markdown_cell(["## 12. 테스트 세트 평가"]))
cells.append(create_code_cell([
    "# Best model 로드\n",
    "model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'best_model.pth')))\n",
    "\n",
    "# 테스트\n",
    "test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)\n",
    "\n",
    "print(\"=\"*70)\n",
    "print(f\"Test Loss: {test_loss:.4f}\")\n",
    "print(f\"Test Accuracy: {test_acc*100:.2f}%\")\n",
    "print(\"=\"*70)\n",
    "\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(test_labels, test_preds, target_names=['Life Noise', 'Sneeze'], digits=4))\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(test_labels, test_preds)\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.imshow(cm, cmap='Blues')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.colorbar()\n",
    "plt.xticks([0, 1], ['Life Noise', 'Sneeze'])\n",
    "plt.yticks([0, 1], ['Life Noise', 'Sneeze'])\n",
    "for i in range(2):\n",
    "    for j in range(2):\n",
    "        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('True')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# 13. 모델 저장
cells.append(create_markdown_cell(["## 13. 모델 Export"]))
cells.append(create_code_cell([
    "# TorchScript\n",
    "model.eval()\n",
    "example = torch.randn(1, 1, 60, 63).to(device)\n",
    "traced = torch.jit.trace(model, example)\n",
    "traced.save(os.path.join(MODEL_SAVE_PATH, 'sneeze_model_scripted.pt'))\n",
    "\n",
    "# ONNX\n",
    "torch.onnx.export(\n",
    "    model, example,\n",
    "    os.path.join(MODEL_SAVE_PATH, 'sneeze_model.onnx'),\n",
    "    input_names=['input'], output_names=['output'],\n",
    "    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}\n",
    ")\n",
    "\n",
    "print(\"모델 저장 완료!\")\n",
    "print(f\"  - PyTorch: best_model.pth\")\n",
    "print(f\"  - TorchScript: sneeze_model_scripted.pt\")\n",
    "print(f\"  - ONNX: sneeze_model.onnx\")"
]))

# 14. 추론 예제
cells.append(create_markdown_cell(["## 14. 추론 예제"]))
cells.append(create_code_cell([
    "def predict_audio(model, audio_path, device):\n",
    "    model.eval()\n",
    "    audio = load_audio(audio_path)\n",
    "    audio = preprocess_audio(audio)\n",
    "    mfcc = extract_mfcc_features(audio)\n",
    "    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(device)\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        output = model(mfcc_tensor)\n",
    "        prob = F.softmax(output, dim=1)\n",
    "        pred = torch.argmax(prob, dim=1).item()\n",
    "        conf = prob[0][pred].item()\n",
    "    \n",
    "    return pred, conf\n",
    "\n",
    "# 테스트\n",
    "test_file = os.path.join(SNEEZE_AUDIO_PATH, os.listdir(SNEEZE_AUDIO_PATH)[0])\n",
    "if test_file.endswith('.wav'):\n",
    "    pred, conf = predict_audio(model, test_file, device)\n",
    "    label = \"Sneeze\" if pred == 1 else \"Life Noise\"\n",
    "    print(f\"파일: {os.path.basename(test_file)}\")\n",
    "    print(f\"예측: {label} (신뢰도: {conf*100:.2f}%)\")"
]))

# 노트북 생성
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# JSON 파일로 저장
output_path = "/Users/bahk_insung/Documents/Github/sneeze_detection/sneeze_detection_lightweight.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"노트북 생성 완료: {output_path}")
