import os
import argparse
import torch
from tqdm import tqdm
from models import tPCN
from get_data import get_dataloaders

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")
    
    total_loss = 0
    correct = 0
    cnt = 0  # 처리한 데이터 개수

    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, pc_loss = model(data)

        # 통계치 계산
        test_loss += pc_loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        cnt += data.size(0) # 현재 배치 사이즈만큼 더함

        # pbar 설명 업데이트 (실시간 표시)
        pbar.set_description(
            f"Epoch [{epoch}] Train Loss: {total_loss/len(pbar.container):.4f}, Acc: {100. * correct/cnt:.2f}%"
        )

def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784, 1)
            
            output, pc_loss = model(data)
            test_loss += pc_loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")

args = argparse.Namespace()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.z_init = 'ff'
args.activation = 'relu'
args.eta = 0.2
args.T = 2 # Number of inference steps
args.data_dir = os.path.expanduser('~/datasets')
args.epochs = 100
args.lr = 1e-4
args.weight_decay = 5e-4
args.num_classes = 10
args.num_workers = 4
args.optimizer = 'adamw'

args.hidden_shape =
args.input_shape = 

model = tPCN(args).to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_loader, test_loader = get_dataloaders('smnist')

for epoch in range(1, args.epochs + 1):
    train(model, args.device, train_loader, optimizer, epoch)
    test(model, args.device, test_loader)