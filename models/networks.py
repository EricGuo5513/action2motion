import torch
import torch.nn as nn


# apply a RNN to model a sequence of trajectory, benificial for global velocity estimation for a whole sequence
# but more parameters
class VelocityNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(VelocityNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = int(hidden_size/4)
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.embed = nn.Linear(input_size, self.hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(self.hidden_size, self.hidden_size) for i in range(self.n_layers)])
        self.linear = nn.Linear(self.hidden_size, output_size)
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
        h_1 = torch.relu(h_1)
        h_2 = self.linear2(h_1)
        h_2 = torch.relu(h_2)
        output = self.linear3(h_2)
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


class HierarchicalDenseLayer(nn.Module):
    def __init__(self, context_size, chains, num_joints, do_all_parent=False):
        super(HierarchicalDenseLayer, self).__init__()
        self._kinematic_chains = chains

        self.PI = 3.1415926

        self.context_size = context_size
        self.do_all_parent = do_all_parent
        self.num_joints = num_joints
        self.construct_net()

    def construct_net(self):
        linear_list = [None] * self.num_joints
        linear_list[0] = nn.Linear(self.context_size, 3)
        for chain in self._kinematic_chains:
            for j in range(1, len(chain)):
                if self.do_all_parent:
                    linear_list[chain[j]] = nn.Linear(self.context_size + j * 3, 3)
                else:
                    linear_list[chain[j]] = nn.Linear(self.context_size + 3, 3)
        self.linears = nn.ModuleList(linear_list)

    def forward(self, context_vec):
        joint_list = [None] * self.num_joints
        # root joint
        outputs = self.linears[0](context_vec)
        joint_list[0] = torch.tanh(outputs) * self.PI

        for chain in self._kinematic_chains:
            parent_preds = [context_vec, joint_list[chain[0]]]
            for j in range(1, len(chain)):
                if self.do_all_parent:
                    inputs = torch.cat(parent_preds, dim=-1)
                else:
                    inputs = torch.cat([context_vec, joint_list[chain[j-1]]], dim=-1)

                outputs = self.linears[chain[j]](inputs)
                joint_list[chain[j]] = torch.tanh(outputs) * self.PI
                parent_preds.append(joint_list[chain[j]])
        pose_vec = torch.cat(joint_list, dim=-1)
        return pose_vec