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

train_ds = datasets.MovingMNIST(args.data_dir, train=True, download=True, transform=train_transform)
test_ds = datasets.MovingMNIST(args.data_dir, train=False, download=True, transform=test_transform)
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
            'identity': nn.Identity()
        }
        activation_layer = act_dict.get(args.activation.lower(), nn.ReLU())

        # Input/Sensory Prediction: z_{k} -> x_{k}
        self.input_predictor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 10),
            activation_layer
        )
        
        # Temporal Prediction: z_{k-1} -> z_{k}
        self.temporal_predictor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 10),
            activation_layer
        )

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

    def inference_step(self, z_t, z_prev, x, k, t, optimizer):
        """Computes gradients for a single inference step

        Args:
            z_t (torch.Tensor): Current values for each layer at iteration t.
            x (torch.Tensor): Input data (Batch Size, Sequence length, 3, 32, 32).
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
            z_state_t = z_t.clone().detach().requires_grad_(True)  # Copy of state
            z_0_t = z_t.clone().detach().requires_grad_(True)  # Copy of initial state

            x_pred_t = self.input_predictor(z_0_t)   
            z_pred_t = self.temporal_predictor(z_0_t)

            # 1. Input prediction loss
            pc_loss = torch.sum((x_pred_t - x)**2) # Input prediction loss

            # 2. Temporal prediction error
            if k > 0:
                z_pred_t = self.temporal_predictor(z_prev)
                pc_loss += torch.sum((z_pred_t)**2)

            pc_loss.backward() # Compute gradients

            delta_zs_t = - z_state_t.grad * self.args.eta - z_0_t.grad

        if t == self.args.T - 1:    # When at last inference step, updates weights (??why here??)
            optimizer.step()

        return delta_zs_t, pc_loss

    def forward_training(self, seq, optimizer):
        '''Performs the entire iterative inference for current sequence
        '''
        preds = []
        pc_loss_list = []

        batch_size = seq.size(0)
        seq_len = seq.size(1)
        z_k = torch.zeros(batch_size, self.hidden_size).to(self.args.device) # Initialized as 0

        for k in range(seq_len):
            x_k = seq[:, k, :]
            z_k_t = z_k.detach().requires_grad_(True) 
            pc_loss_list.append([])
            for t in range(self.args.T):    # Iterates for T inference steps
                delta_z_t, pc_loss = self.inference_step(z_k_t, x_k, t, optimizer)
                z_k_t += delta_z_t   # Updates states
                pc_loss_list[k].append(pc_loss.item())

            preds.append(self.input_predictor(z_k_t))    # Use conversed hidden state to make final prediction
        return {
            'preds': preds, 
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
    '''TODO - rewrite for sequential
    '''
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
