import torch
import random
import numpy as np
from torch import nn
from typing import Union

from constant import DEVICE, OUTPUT_KEYS_TO_INDEX
from utils import from_numpy, to_numpy, normalize, unnormalize


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(inplace=True),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(inplace=True),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(inplace=True),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = 'tanh',
    output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.BatchNorm1d(size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


class AGCModel(nn.Module):

    round_keys = [
        "comp1.Scr1.Pos",
        "comp1.Scr2.Pos",
    ]
    round_idx = [OUTPUT_KEYS_TO_INDEX[key] for key in round_keys]

    def __init__(self, cp_dim, ep_dim, op_dim, norm_data):
        super(AGCModel, self).__init__()

        self.cp_dim = cp_dim
        self.ep_dim = ep_dim
        self.op_dim = op_dim
        self.norm_data = norm_data  # this contains numpy arrays

        self.delta_network = build_mlp(
            input_size=self.cp_dim + self.ep_dim + self.op_dim,
            output_size=self.op_dim,
            n_layers=3,
            size=64,
            activation='leaky_relu'
        )
        self.delta_network.to(DEVICE)
        self.criterion = nn.MSELoss()

        self.cp_mean = from_numpy(norm_data['cp_mean'])
        self.ep_mean = from_numpy(norm_data['ep_mean'])
        self.op_mean = from_numpy(norm_data['op_mean'])
        self.delta_mean = from_numpy(norm_data['delta_mean'])
        self.cp_std = from_numpy(norm_data['cp_std'])
        self.ep_std = from_numpy(norm_data['ep_std'])
        self.op_std = from_numpy(norm_data['op_std'])
        self.delta_std = from_numpy(norm_data['delta_std'])

    def forward(self, cp, ep, op):  # this should only be used for training
        """
            param `cp`: Unnormalized actions -> torch.Tensor (B x CP_DIM)
            param `ep`: Unnormalized weather observations -> torch.Tensor (B x EP_DIM)
            param `op`: Unnormalized greenhouse observations -> torch.Tensor (B x OP_DIM)
            return `delta_pred_normalized`: the normalized (i.e. not unnormalized) 
            output of the delta network -> torch.Tensor (B x OP_DIM)
        """
        cp = cp.to(DEVICE)
        ep = ep.to(DEVICE)
        op = op.to(DEVICE)

        # normalize input data to mean 0, std 1
        cp_normalized = normalize(cp, self.cp_mean, self.cp_std)
        ep_normalized = normalize(ep, self.ep_mean, self.ep_std)
        op_normalized = normalize(op, self.op_mean, self.op_std)

        # predicted change in op
        input = torch.cat([cp_normalized, ep_normalized, op_normalized], dim=1)

        return self.delta_network(input)

    def predict_op(self, cp, ep, op):  # use this only for prediction after training
        """
            param `cp`: Unnormalized control actions `a_cp[t+1]` -> numpy.ndarray (CP_DIM)
            param `ep`: Unnormalized weather observations `s_ep[t]` -> numpy.ndarray (EP_DIM)
            param `op`: Unnormalized greenhouse observations `s_op[t]` -> numpy.ndarray (OP_DIM)
            return `op_next_pred`: the predicted `s_op[t+1]` -> numpy.ndarray
        """
        cp_tensor = from_numpy(cp).unsqueeze(0)
        ep_tensor = from_numpy(ep).unsqueeze(0)
        op_tensor = from_numpy(op).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            delta_pred_normalized = self.forward(cp_tensor, ep_tensor, op_tensor)
            delta_pred = unnormalize(delta_pred_normalized, self.delta_mean, self.delta_std)
            op_next_pred = torch.relu(op_tensor + delta_pred)  # force positive
            op_next_pred[:,self.round_idx] = torch.round(op_next_pred[:,self.round_idx])  # force 0/1 on Scr.Pos

        return to_numpy(op_next_pred).flatten()

    def loss(self, input, target):
        """
            param `input`: Unnormalized (`cp`, `ep`, `op`) -> torch.Tensor
            param `target`: Unnormalized `op_next` -> torch.Tensor
            return `loss`: MSE Loss between normalized `op_delta` and `net(cp,ep,op)` -> torch.Tensor
        """
        cp, ep, op = input
        op_next = target

        cp = cp.to(DEVICE)
        ep = ep.to(DEVICE)
        op = op.to(DEVICE)
        op_next = op_next.to(DEVICE)

        delta_real_normalized = normalize(op_next - op, self.delta_mean, self.delta_std)
        delta_pred_normalized = self.forward(cp, ep, op)
        return self.criterion(delta_pred_normalized, delta_real_normalized)

    def rollout(self, cp_all, ep_all, op_all, lookahead=1):
        """
            param `cp_all`: Unnormalized control actions `a_cp[t+1:t+1+T]` -> numpy.ndarray (T x CP_DIM)
            param `ep_all`: Unnormalized weather observations `s_ep[t:t+T]` -> numpy.ndarray (T x EP_DIM)
            param `op_all`: Unnormalized greenhouse observations `s_op[t:t+T]` -> numpy.ndarray (OP_DIM)
            return `op_next_all_pred`: Predicted `s_op[t+lookahead:t+1+T]` -> numpy.ndarray ((T-lookahead+1) x OP_DIM)
        """
        op_next_all_pred = []
        T = cp_all.shape[0]

        for t in range(T-lookahead+1):
            op = op_all[t]
            for s in range(lookahead):
                cp = cp_all[t+s]
                ep = ep_all[t+s]
                op = self.predict_op(cp, ep, op)
            op_next_all_pred.append(op)
        
        return np.asarray(op_next_all_pred)
    
    
    def rollout_one_trajectory(self, cp_all, ep_all, op_init):
        """
            param `cp_all`: Unnormalized control actions `a_cp[t:t+T]` -> numpy.ndarray (T x CP_DIM)
            param `ep_all`: Unnormalized weather observations `s_ep[t:t+T]` -> numpy.ndarray (T x EP_DIM)
            param `op_all`: Unnormalized greenhouse observations `s_op[t:t+T]` -> numpy.ndarray (OP_DIM)
            return `op_next_all_pred`: Predicted `s_op[t+1:t+1+T]` -> numpy.ndarray (T x OP_DIM)
        """
        T = cp_all.shape[0]
        op = op_init
        op_next_all_pred = []

        for t in range(T):
            cp = cp_all[t]
            ep = ep_all[t]
            op = self.predict_op(cp, ep, op)
            op_next_all_pred.append(op)
        
        return np.asarray(op_next_all_pred)


class AGCModelEnsemble(nn.Module):
    def __init__(self, cp_dim, ep_dim, op_dim, ckpt_paths):
        super(AGCModelEnsemble, self).__init__()

        models = nn.ModuleList()
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path)
            model = AGCModel(cp_dim, ep_dim, op_dim, ckpt['norm_data'])
            model.load_state_dict(ckpt['state_dict'])
            models.append(model)
        self.child_models = models
        self.num_childmodels = len(self.child_models)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, cp, ep, op, mode='average'):
        """
            param `cp`: Unnormalized control action `a_cp[t+1]` -> numpy.ndarray
            param `ep`: Unnormalized weather observation `s_ep[t]` -> numpy.ndarray
            param `op`: Unnormalized greenhouse observation `s_op[t]` -> numpy.ndarray
            return `op_next_pred`: the predicted `s_op[t+1]` -> numpy.ndarray
        """
        cp = cp.to(DEVICE)
        ep = ep.to(DEVICE)
        op = op.to(DEVICE)

        if mode == 'average':
            op_next_pred_all = [model.predict_op(cp, ep, op) for model in self.child_models]
            op_next_pred = np.mean(op_next_pred_all, axis=0)
        elif mode == 'stochastic':
            model = random.choice(self.child_models)
            op_next_pred = model.predict_op(cp, ep, op)
        else:
            raise NotImplementedError

        return op_next_pred

    def rollout(self, cp_all, ep_all, op_all, lookahead=1, mode='average'):
        """
            param `cp_all`: Unnormalized control actions `a_cp[t+1:t+1+T]` -> numpy.ndarray (T x CP_DIM)
            param `ep_all`: Unnormalized weather observations `s_ep[t:t+T]` -> numpy.ndarray (T x EP_DIM)
            param `op_all`: Unnormalized greenhouse observations `s_op[t:t+T]` -> numpy.ndarray (OP_DIM)
            return `op_next_all_pred`: Predicted `s_op[t+lookahead:t+1+T]` -> numpy.ndarray ((T-lookahead+1) x OP_DIM)
        """
        op_next_all_pred = []
        T = cp_all.shape[0]

        for t in range(T-lookahead+1):
            op = op_all[t]
            for s in range(lookahead):
                cp = cp_all[t+s]
                ep = ep_all[t+s]
                op = self.forward(cp, ep, op, mode=mode)
            op_next_all_pred.append(op)
        
        return np.asarray(op_next_all_pred)

    def rollout_one_trajectory(self, cp_all, ep_all, op_init, mode='average'):
        """
            param `cp_all`: Unnormalized control actions `a_cp[t:t+T]` -> numpy.ndarray (T x CP_DIM)
            param `ep_all`: Unnormalized weather observations `s_ep[t:t+T]` -> numpy.ndarray (T x EP_DIM)
            param `op_all`: Unnormalized greenhouse observations `s_op[t:t+T]` -> numpy.ndarray (OP_DIM)
            return `op_next_all_pred`: Predicted `s_op[t+1:t+1+T]` -> numpy.ndarray (T x OP_DIM)
        """
        T = cp_all.shape[0]
        op = op_init
        op_next_all_pred = []

        for t in range(T):
            cp = cp_all[t]
            ep = ep_all[t]
            op = self.forward(cp, ep, op, mode=mode)
            op_next_all_pred.append(op)
        
        return np.asarray(op_next_all_pred)


if __name__ == '__main__':
    import numpy as np

    CP_DIM = 56
    EP_DIM = 5
    OP_DIM = 20

    cp = np.random.rand(CP_DIM)
    ep = np.random.rand(EP_DIM)
    op = np.random.rand(OP_DIM)
    op_next = np.random.rand(OP_DIM)

    norm_data = {
        'cp_mean': np.random.rand(CP_DIM), 'cp_std': np.random.randint(2, size=CP_DIM),
        'ep_mean': np.random.rand(EP_DIM), 'ep_std': np.random.randint(2, size=EP_DIM),
        'op_mean': np.random.rand(OP_DIM), 'op_std': np.random.randint(2, size=OP_DIM),
        'delta_mean': np.random.rand(OP_DIM), 'delta_std': np.random.randint(2, size=OP_DIM),
    }

    model = AGCModel(CP_DIM, EP_DIM, OP_DIM, norm_data)

    op_next_pred = model.predict_op(cp, ep, op)

    print(model.delta_network)
    print(type(op_next_pred), op_next_pred)

    BS = 10
    cp_tensor = torch.rand(BS, CP_DIM)
    ep_tensor = torch.rand(BS, EP_DIM)
    op_tensor = torch.rand(BS, OP_DIM)
    op_next_tensor = torch.rand(BS, OP_DIM)
    op_next_pred_tensor = model(cp_tensor, ep_tensor, op_tensor)

    print(type(op_next_pred_tensor), op_next_pred_tensor)

    loss = model.loss((cp_tensor, ep_tensor, op_tensor), op_next_tensor)
    
    print(loss.item())
