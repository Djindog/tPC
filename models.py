import os
import time
import torch
import torch.nn.functional as F
from torch import nn

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
            nn.Linear(args.hidden_shape, args.input_shape),
            activation_layer
        )
        
        # Temporal Prediction: z_{k-1} -> z_{k}
        self.temporal_predictor = nn.Sequential(
            nn.Linear(args.hidden_shape, args.hidden_shape),
            activation_layer
        )

        # Classifier: z_{k} -> y
        self.classifier = nn.Linear(args.hidden_shape, 10)

    def forward(self, z_prev):
        z_pred = self.temporal_predictor(z_prev)
        x_pred = self.input_predictor(z_pred) 
                
        return z_pred, x_pred
    
    def inference(self, x_k, z_prev, optimizer):
        z = self.temporal_predictor(z_prev)

        for t in range(self.args.T):
            with torch.enable_grad():
                z = z.clone().detach().requires_grad_(True)

                z_pred = self.temporal_predictor(z_prev)
                x_pred = self.input_predictor(z)   

                # 1. Input prediction loss
                pc_loss = torch.sum((x_k - x_pred)**2)

                # 2. Temporal prediction error
                pc_loss += torch.sum((z - z_pred)**2)

                # pc_loss_list[k].append(pc_loss.item()) # Log loss

                pc_loss.backward() # Compute gradients

            with torch.no_grad():
                z = z - self.args.eta * z.grad
                z_prev = z.detach() # Fixed prediction assumption?

        return 
        
    
    def forward_train(self, data, optimizer):
        '''Performs the entire iterative inference for current sequence
        '''
        preds = []
        pc_loss_list = []

        batch_size = data.size(0)
        seq_len = data.size(1)

        z = torch.zeros(batch_size, self.args.hidden_shape).to(self.args.device) # Initialized as 0
        z_prev = z.clone().detach()

        for k in range(seq_len):
            x_k = data[:, k, :]
            pc_loss_list.append([])

            # Iterates for T inference steps
            for t in range(self.args.T):
                optimizer.zero_grad()
                with torch.enable_grad():
                    z = z.clone().detach().requires_grad_(True)

                    z_pred = self.temporal_predictor(z_prev)
                    x_pred = self.input_predictor(z_pred)   

                    # 1. Input prediction loss
                    pc_loss = torch.sum((x_k - x_pred)**2)

                    # 2. Temporal prediction error
                    pc_loss += torch.sum((z - z_pred)**2)

                    pc_loss_list[k].append(pc_loss.item()) # Log loss

                    pc_loss.backward() # Compute gradients

                with torch.no_grad():
                    z = z - self.args.eta * z.grad
                    z_prev = z.detach() # Fixed prediction assumption?

            preds.append(x_pred)    # Log final prediction
            with torch.no_grad():   # Initialize next inference with predicted z
                z = z_pred
                

            optimizer.step() # Weight updates

        return {
            'preds': preds, 
            'pc_loss': pc_loss,
            'pc_loss_list': pc_loss_list    # Log of all pc_loss
        }