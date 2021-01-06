import torch
import torch.nn as nn
import random

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")
class ConditionedRNN(nn.Module):
    def __init__(self, n_categeories, input_size, hidden_size, output_size):
        super(ConditionedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear_c2e = nn.Linear(n_categeories, input_size)
        self.gru = nn.GRU(input_size * 2, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)

    def forward(self, pose_unit, hidden_unit, category_tensor):
        pose_unit = pose_unit.view(1, 1, -1)
        category_unit = self.linear_c2e(category_tensor).view(1, 1, -1)
        input_unit = torch.cat((pose_unit, category_unit), axis=2)
        # print(input_unit.size(), hidden_unit.size())
        gru_output, gru_hidden = self.gru(input_unit, hidden_unit)
        output = self.linear_output(gru_output)
        return output, gru_hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train_conditionedRNN(category_tensor, input_line_tensor, target_line_tensor,
                         model, model_optimizer, criterion):
    model_optimizer.zero_grad()

    loss = 0
    length = target_line_tensor.size(0)
    teacher_forcing_ratio = 0.5
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    model_hidden = model.initHidden()
    model_input = input_line_tensor[0]

    if use_teacher_forcing:
        for di in range(length):
            model_output, model_hidden = model(
                model_input, model_hidden, category_tensor
            )
            loss += criterion(model_output, target_line_tensor[di])
            model_input = target_line_tensor[di]
    else:
        for di in range(length):
            model_output, model_hidden = model(
                model_input, model_hidden, category_tensor
            )
            loss += criterion(model_output, target_line_tensor[di])
            model_input = model_output

    loss.backward()
    model_optimizer.step()
    return loss.item() / length
