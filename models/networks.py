import torch
import torch.nn as nn


# apply a RNN to model a sequence of trajectory, benificial for global velocity estimation for a whole sequence
# but more parameters
class VelocityNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(VelocityNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.linear = nn.Linear(hidden_size, output_size)
        self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.gru[i](h_in, self.hidden[i])
            h_in = self.hidden[i]
        output = self.linear(h_in)
        output = nn.LeakyReLU(negative_slope=0.1)(output)
        return output


# Simplied version of velocity estimation, which is only for consecutive two frames.
# Less parameters, basically a MLP model
class VelocityNetwork_Sim(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(VelocityNetwork_Sim, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear3 = nn.Linear(int(hidden_size/2), output_size)

    def init_hidden(self, num_samples=None):
        pass

    def forward(self, inputs):
        h_1 = self.linear1(inputs)
        h_1 = nn.LeakyReLU(negative_slope=0.1)(h_1)
        h_2 = self.linear2(h_1)
        h_2 = nn.LeakyReLU(negative_slope=0.1)(h_2)
        h_3 = self.linear3(h_2)
        output = nn.LeakyReLU(negative_slope=0.1)(h_3)
        return output


class VelocityNetworkHierarchy(nn.Module):
    def __init__(self, output_size, chains):
        super(VelocityNetworkHierarchy, self).__init__()
        self.output_size = output_size
        self.chains = chains
        self.inter_linear = nn.ModuleList([nn.Linear(len(chain), 5) for chain in chains])
        self.linear = nn.Linear(5*len(chains) + 10, 30)
        self.out_linear = nn.Linear(30, 3)

    def init_hidden(self, num_samples=None):
        pass

    def forward(self, inputs):
        p1, p2, hid = inputs
        res_pose = p1 - p2
        h_vec = None
        for i in range(len(self.chains)):
            chain_in = res_pose[:, self.chains[i]]
            chain_out = self.inter_linear[i](chain_in)
            chain_out = nn.LeakyReLU(negative_slope=0.1)(chain_out)
            if h_vec is None:
                h_vec = chain_out
            else:
                h_vec = torch.cat((h_vec, chain_out), dim=-1)
        h_in = torch.cat((h_vec, hid), dim=-1)
        h = self.linear(h_in)
        h = nn.LeakyReLU(negative_slope=0.1)(h)
        output = self.out_linear(h)
        return output