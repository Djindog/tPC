import os
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

args = argparse.Namespace()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.z_init = 'ff'
args.update_rule = 'pcn'
args.backbone = 'vgg13'
args.activation = 'relu'
args.eta = 0.2
args.T = 20 # Number of inference steps
args.data_dir = os.path.expanduser('~/datasets')
args.batch_size = 128
args.epochs = 100
args.lr = 1e-4
args.weight_decay = 5e-4
args.num_classes = 10
args.num_workers = 4
args.optimizer = 'adamw'
args.dataset = 'CIFAR10'
if args.dataset == 'MNIST':
    args.num_classes = 10
    args.img_shape = (1, 28, 28)
elif args.dataset == 'CIFAR10':
    args.num_classes = 10
    args.img_shape = (3, 32, 32)
else:
    raise ValueError(f"Dataset {args.dataset} not supported")
args.mean = [0.4914, 0.4822, 0.4465]
args.std = [0.2023, 0.1994, 0.2010]

now = time.strftime('%Y%m%d-%H%M%S')
args.log_dir = os.path.join('log', now)
os.makedirs(args.log_dir, exist_ok=True)
log_file = os.path.join(args.log_dir, 'metrics.log')

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(args.mean, args.std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(args.mean, args.std),
])

train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

class tPCN(nn.Module):
    def __init__(self, args):
        super(tPCN, self).__init__()
        self.args = args

        act_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity() # 아무 연산도 하지 않음 (Linear만 남음)
        }
        activation_layer = act_dict.get(args.activation.lower(), nn.ReLU())

        self.input_predictor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 10),
            activation_layer)
        
        self.temporal_predictor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 10),
            activation_layer)

    def forward(self, x, y, optimizer):
        if self.args.z_init == 'ff':
            return self.forward_ff(x, y, optimizer)
        else:
            raise ValueError(f"z_init {self.args.z_init} not supported")

    def predict_forward(self, x):
        '''Performs feed-forward pass to compute output
        '''
        r = x
        for backbone_module in self.backbone_module_list:
            r = backbone_module(r)
        return {
            'pred': r,
        }

    def forward_train(self, zs_t, zs_0, x, y, k, t, optimizer):
        """Computes gradients for a single inference step

        Args:
            zs_t (list[torch.Tensor]): Current values for each layer at iteration t.
            zs_0 (list[torch.Tensor]): Initial values for each layer.
            x (torch.Tensor): Input data (Batch Size, 3, 32, 32).
            y (torch.Tensor): Target labels (Batch Size).
            k (int): Current input index.
            t (int): Current inference step (0 ~ T-1).
            optimizer (torch.optim.Optimizer): optimizer for model weight updates.

        Returns:
            tuple: (delta_zs_t, pc_loss)
                - delta_zs_t (list[torch.Tensor]): 에러를 줄이기 위해 계산된 각 레이어별 뉴런 수정량.
                - pc_loss (torch.Tensor): 현재 단계에서의 전체 예측 오차 합계.
        """
        optimizer.zero_grad()
        with torch.enable_grad():
            zs_state_t = [z.clone().detach().requires_grad_(True) for z in zs_t]    # Copy of state
            zs_0_t = [z.clone().detach().requires_grad_(True) for z in zs_0]    # Copy of initial state
            zs_0_pred_t = []    # Predictions
            for idx, backbone_module in enumerate(self.backbone_module_list):
                zs_0_pred_t.append(backbone_module(zs_0_t[idx]))    # Generates predictions of next layer with current layer

            pc_loss = torch.sum((zs_state_t[0] - zs_0_t[0])**2) # Input layer loss
            for idx, backbone_module in enumerate(self.backbone_module_list):
                if idx != len(self.backbone_module_list) - 1:
                    pc_loss += torch.sum((zs_state_t[idx+1] - zs_0_pred_t[idx])**2) # Prediction loss for each layer
                else:
                    pc_loss += torch.nn.functional.cross_entropy(zs_state_t[-1], y)     # Classification loss
                    pc_loss += torch.nn.functional.cross_entropy(zs_0_pred_t[-1], y)    # Prediction loss for last layer

            pc_loss.backward() # Compute gradients

            delta_zs_t = [torch.zeros_like(z) for z in zs_state_t]
            for idx in range(1, len(zs_state_t)):
                delta_zs_t[idx] = - zs_state_t[idx].grad * self.args.eta
                if idx != len(zs_state_t) - 1:
                    delta_zs_t[idx] += - zs_0_t[idx].grad * self.args.eta

        if t == self.args.T - 1:    # When at last inference step, updates weights (??why here??)
            optimizer.step()

        return delta_zs_t, pc_loss

    def forward_ff(self, x, y, optimizer):
        '''Performs the entire iterative inference stage
        '''
        zs_t = self.init_zs_ff(x, y)
        zs_0 = [z.clone().detach().requires_grad_(True) for z in zs_t]
        pc_loss_list = []
        for t in range(self.args.T):    # Iterates for T inference steps
            delta_zs_t, pc_loss = self.forward_train(zs_t, zs_0, x, y, t, optimizer)
            zs_t = [z + delta_z for z, delta_z in zip(zs_t, delta_zs_t)]    # updates states
            pc_loss_list.append(pc_loss.item())
        return {
            'pred': zs_t[-1], 
            'pc_loss': pc_loss,
            'pc_loss_list': pc_loss_list    # Log of all pc_loss
        }

    def init_zs_ff(self, x, y):
        '''Feed-forward initialization for all layers.
        '''
        zs = [x]
        for backbone_module in self.backbone_module_list:
            r = backbone_module(zs[-1])
            zs.append(r)
        return zs

pcn = tPCN(args).to(args.device)
optimizer = torch.optim.AdamW(pcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_losses, test_losses = [], []
train_accs, test_accs = [], []

def train_one_epoch():
    pcn.train()
    total_loss = 0
    correct = 0
    correct_forward = 0
    cnt = 0
    pbar = tqdm(train_loader)
    for x, y in pbar:
        x, y = x.to(args.device), y.to(args.device)

        pred_forward = pcn.predict_forward(x)['pred'].argmax(dim=1)
        correct_forward += (pred_forward == y).sum().item()

        output_dict = pcn(x, y, optimizer)
        loss = output_dict['pc_loss']
        total_loss += loss.item() * x.size(0)
        preds = output_dict['pred'].argmax(dim=1)
        correct += (preds == y).sum().item()
        cnt += x.size(0)

        pbar.set_description(f"Train Loss: {total_loss/cnt:.4f}, Acc: {correct/cnt:.4f}, Acc_forward: {correct_forward/cnt:.4f}")

    return total_loss / len(train_ds), correct / len(train_ds)

def eval_one_epoch():
    pcn.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)

            output_dict = pcn.predict_forward(x)
            total_loss += 0
            preds = output_dict['pred'].argmax(dim=1)
            correct += (preds == y).sum().item()

    return total_loss / len(test_ds), correct / len(test_ds)

for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = train_one_epoch()
    te_loss, te_acc = eval_one_epoch()

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    with open(log_file, 'a') as f:
        f.write(f"{epoch}\t{tr_loss:.4f}\t{tr_acc:.4f}\t{te_loss:.4f}\t{te_acc:.4f}\n")

    epochs = list(range(1, epoch+1))
    plt.figure(); plt.plot(epochs, train_losses, label='train'); plt.plot(epochs, test_losses, label='test')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(args.log_dir, 'loss.png')); plt.close()

    plt.figure(); plt.plot(epochs, train_accs, label='train'); plt.plot(epochs, test_accs, label='test')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(os.path.join(args.log_dir, 'acc.png')); plt.close()

    print(f"Epoch {epoch:03d} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}")
