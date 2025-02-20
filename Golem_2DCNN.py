#!/usr/bin/env python3

import sklearn.model_selection
import torch
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Final
import itertools
import random
import librosa.feature
import sklearn.preprocessing
from torch import nn
from torch.utils.data import DataLoader
import typing
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

type SampleRate = int
type Samples = np.ndarray
type WavData = tuple[Samples, SampleRate]
type Label = str
type Spectrogram = np.ndarray
type Data = list[tuple[Label, Spectrogram]]

MEAN_STD_STORE_PATH: Final = Path("normalization.txt")
MEAN_STD: Final[tuple[float, float]] = typing.cast(tuple[float, float], tuple([float(x) for x in MEAN_STD_STORE_PATH.read_text().split(maxsplit=1)]))
DEVICE: Final = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_SET_DIR: Final = Path("train/audio")
NETWORK_INPUT: Final = (128, 50)
COMMANDS: Final[list[Label]] = [
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "four",
    "go",
    "happy",
    "house",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "wow",
    "yes",
    "zero",
    #"unknown"
]

def generate_spectrograms(data: list[tuple[Label, WavData, Path]]) -> list[tuple[Label, Spectrogram, Path]]:
    def one_spectrogram(data: WavData) -> Spectrogram:
        mel_signal = librosa.feature.melspectrogram(y=data[0], sr=data[1], hop_length=int((len(data[0]))/NETWORK_INPUT[1])+1, n_mels=NETWORK_INPUT[0])
        spectrogram = np.abs(mel_signal)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db

    return list(map(lambda x: (x[0], one_spectrogram(x[1]), x[2]), data))

def normalize_spectrograms(data: list[tuple[Label, Spectrogram, Path]]) -> Data:
    labels: list[Label]
    spectrograms: list[Spectrogram]
    labels, spectrograms, _ = typing.cast(tuple[list[Label], list[Spectrogram], list[Path]], zip(*data))
    print(f"{np.mean(spectrograms):+.15f}\n{np.std(spectrograms):+.15f}")
    print(f"{np.mean(spectrograms):+.15f}\n{np.std(spectrograms):+.15f}", file=MEAN_STD_STORE_PATH.open('w'))
    exit(0)
    spectrograms = typing.cast(list[Spectrogram], sklearn.preprocessing.scale([s.flatten() for s in spectrograms]))
    spectrograms = [s.reshape(NETWORK_INPUT) for s in spectrograms]
    return list(zip(labels, spectrograms))

def is_cached(filepath: Path) -> bool:
    return ("cache" / filepath.with_suffix(".npy")).exists()

def load_from_cache(files: list[tuple[Label, Path]]) -> list[tuple[Label, Spectrogram]]:
    return [(label, np.load("cache" / filepath.with_suffix(".npy"))) for label, filepath in files]

def split_on_cache(files: list[tuple[Label, Path]]) -> tuple[list[tuple[Label, Path]], list[tuple[Label, Path]]]:
    return list(filter(lambda x: is_cached(x[1]), files)), list(filter(lambda x: not is_cached(x[1]), files))

def get_all_filepaths(labels: list[Label]) -> list[tuple[Label, Path]]:
    return [(label, file) for label in labels for file in (TRAINING_SET_DIR / label).glob("*.wav")]

def read_all(files: list[tuple[Label, Path]]) -> list[tuple[Label, WavData, Path]]:
    return [(label, librosa.load(str(path)), path) for label, path in files] # type: ignore

def save_to_cache(data: list[tuple[Label, Spectrogram, Path]]):
    for _, spectrogram, path in data:
        ("cache" / path).parent.mkdir(parents=True, exist_ok=True)
        np.save("cache" / path.with_suffix(".npy"), spectrogram)

def trim_silence(data: list[tuple[Label, WavData, Path]]) -> list[tuple[Label, WavData, Path]]:
    return [(label, (librosa.effects.trim(wave[0])[0], wave[1]), path) for label, wave, path in data]

def generate(files: list[tuple[Label, Path]], normalization_data: tuple[float, float]) -> list[tuple[Label, Spectrogram, Path]]:
    mean, std = normalization_data
    data = read_all(files)
    data = trim_silence(data)
    data = generate_spectrograms(data)
    #normalize_spectrograms(data) # type: ignore
    data = [(label, (spectrogram - mean) / std, path) for label, spectrogram, path in data]
    return data

def fast_load(labels: list[Label]) -> Data:
    files = get_all_filepaths(labels)
    cached, to_generate = split_on_cache(files)
    cached = load_from_cache(cached)
    generated = generate(to_generate, MEAN_STD)
    save_to_cache(generated)
    generated = [(label, data) for label, data, _ in generated]
    return list(itertools.chain(cached, generated))

def train_loop(dataloader: DataLoader, model: nn.Module, batch_size: int, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
    size = len(dataloader.dataset) # type: ignore
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 25 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss {loss:>4f}, [{current:>5d}/{size:>5d}]")

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    size = len(dataloader.dataset) # type: ignore
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Result:\n\tAccuracy: {correct*100:.4f}%\n\tAverage Loss: {test_loss:.6f}")

class AudioCommandClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(11, 11), padding=5),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=10, out_channels=50, kernel_size=(7, 7), padding=3),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(7, 7), padding=3),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(p=0.15),
            nn.Flatten(),
            nn.LazyLinear(500),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(500),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(len(COMMANDS)),
        )
    
    def forward(self, x):
        return self.stack(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def label_to_array(label: str) -> list[float]:
    return [1.0 if label == command else 0.0 for command in COMMANDS]

epochs = 100
batch_size = 128

data_set = fast_load(COMMANDS)
print("data loaded")

train, validate = sklearn.model_selection.train_test_split(data_set, train_size=0.75)

train_tensor_x = torch.Tensor(np.array(list(map(lambda x: [x[1]], train)))).to(DEVICE)
train_tensor_y = torch.Tensor(np.array(list(map(lambda x: label_to_array(x[0]), train)))).to(DEVICE)
train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

validate_tensor_x = torch.Tensor(np.array(list(map(lambda x: [x[1]], validate)))).to(DEVICE)
validate_tensor_y = torch.Tensor(np.array(list(map(lambda x: label_to_array(x[0]), validate)))).to(DEVICE)
validate_dataset = torch.utils.data.TensorDataset(validate_tensor_x, validate_tensor_y)
validate_loader = DataLoader(validate_dataset, batch_size, shuffle=True)

model = AudioCommandClassifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)

for t in range(epochs):
    print(f"Epoch {t+1} ---------------------")
    train_loop(train_loader, model, batch_size, loss_fn, optimizer)
    test_loop(validate_loader, model, loss_fn)
print("done")
