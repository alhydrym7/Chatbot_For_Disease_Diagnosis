import torch
import torch.nn as nn


import torch
import torch.nn as nn

import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.l2 = nn.Linear(hidden_size, hidden_size*2)  # Larger hidden layer
#         self.l3 = nn.Linear(hidden_size*2, hidden_size)  # Another hidden layer
#         self.l4 = nn.Linear(hidden_size, num_classes)  # Output layer
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()  # Use Tanh activation for the output layer
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         out = self.relu(out)
#         out = self.l3(out)
#         out = self.relu(out)
#         out = self.l4(out)
#         out = self.tanh(out)  # Use Tanh activation for the output layer
#         return out




class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conv_out = F.relu(self.conv1d(embedded))
        global_pool = self.global_pooling(conv_out).squeeze(2)
        output = self.fc(global_pool)
        return output

# class LSTMNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(LSTMNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         if len(x.shape) == 2:
#             x = x.unsqueeze(0)  # Add batch dimension if input is unbatched

#         h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

#         out, _ = self.lstm(x, (h0, c0))
#         out = self.relu(out[:, -1, :])
#         out = self.fc(out)

#         if len(out.shape) == 3:
#             out = out.squeeze(0)  # Remove batch dimension if input was unbatched

#         return out




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.lstm(input.view(len(input), 1, -1))
        output = self.fc(output.view(len(input), -1))
        return output



class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state, seq_lengths):
        # Perform the forward pass
        packed_input = nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, enforce_sorted=False)
        packed_output, hidden_state = self.lstm(packed_input, hidden_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        output = self.fc(output)
        return output, hidden_state








# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        out = self.fc(hidden[-1])
        return out