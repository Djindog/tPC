import os
import numpy as np
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class MovingMNISTDataset(Dataset):
    def __init__(self, root_dir='./data', download=True, seq_len=20, train=True):
        self.root_dir = root_dir
        self.filepath = os.path.join(root_dir, 'mnist_test_seq.npy')
        self.seq_len = seq_len
        self.train = train
        
        if download and not os.path.exists(self.filepath):
            os.makedirs(root_dir, exist_ok=True)
            url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
            print("Downloading Moving MNIST...")
            urllib.request.urlretrieve(url, self.filepath)
            print("Downloaded.")
            
        # 데이터 로드: 원본 shape은 (20, 10000, 64, 64) -> (Seq, Batch, H, W)
        data = np.load(self.filepath)
        
        # PyTorch 편의를 위해 shape 변경: (Batch, Seq, H, W) -> (10000, 20, 64, 64)
        data = np.transpose(data, (1, 0, 2, 3))
        
        # Train / Test 분할 (단순 분할: 앞 8000개 Train, 뒤 2000개 Test)
        if self.train:
            self.data = data[:8000, :self.seq_len, :, :]
        else:
            self.data = data[8000:, :self.seq_len, :, :]
            
        # 픽셀 값을 0~1 사이로 정규화
        self.data = self.data.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # shape: (Seq_len, 64, 64) -> (Seq_len, 1(Channel), 64, 64)
        x = self.data[idx]
        x = np.expand_dims(x, axis=1) 
        
        # [주의] 표준 데이터셋에는 라벨이 없으므로, 임의의 클래스(0~9)를 부여합니다.
        # 실제 데이터셋의 라벨이 있다면 이 부분을 수정하세요.
        dummy_label = np.random.randint(0, 10) 
        
        return torch.tensor(x), torch.tensor(dummy_label, dtype=torch.long)

def get_dataloaders(dataset_name, batch_size=64, seq_len=10, root_dir='./data'):
    """데이터셋 이름에 따라 적절한 로더를 반환하는 함수"""
    if dataset_name == 'smnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(28, 28)) # (784, 1) - straightens data
        ])
        train_ds = datasets.MNIST(root_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == 'mmnist':
        train_ds = MovingMNISTDataset(root_dir, seq_len=seq_len, train=True)
        test_ds = MovingMNISTDataset(root_dir, seq_len=seq_len, train=False)
        
    else:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader