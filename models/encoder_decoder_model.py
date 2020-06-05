import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderFC(nn.Module):
    def __init__(self, n_categories, hidden_size):
        super(EncoderFC, self).__init__()
        self.c2e = nn.Linear(n_categories, hidden_size)

    def forward(self, category):
        embeded_ouput = self.c2e(category)
        return embeded_ouput


class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_unit, hidden):
        # print(input_unit.size())
        # print(hidden.size())
        input_unit = input_unit.view(1, 1, -1)
        hidden = hidden.view(1, 1, -1)
        output, hidden = self.gru(input_unit, hidden)
        output = self.linear(output)
        return output, hidden

def train_endecoderModel(category_tensor, input_line_tensor, target_line_tensor,\
                        encoder, decoder, encoder_optimizer, criterion, decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    length = target_line_tensor.size(0)
    teacher_forcing_ratio = 0.5

    encoder_output = encoder(category_tensor)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    decoder_input = input_line_tensor[0]
    # print(encoder_output.size())
    decoder_hidden = encoder_output
    if use_teacher_forcing:
        for di in range(length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            loss += criterion(decoder_output, target_line_tensor[di])
            decoder_input = target_line_tensor[di]
    else:
        for di in range(length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            loss += criterion(decoder_output, target_line_tensor[di])
            decoder_input = decoder_output
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / length
