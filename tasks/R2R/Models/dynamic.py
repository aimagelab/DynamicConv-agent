import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionEncoder(nn.Module):
    """ Encodes instruction via LSTM """
    def __init__(self, input_size=300, hidden_size=512, use_bias=True):
        super(InstructionEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        """ LSTM init"""
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias=self.use_bias)
        """ init weights"""
        for name, param in self.lstm_cell.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        """ Checking data shape """
        forwd = x
        assert forwd.shape[1] == self.input_size, "Expected input with shape [batch, %s, seq_len], found %s" % (self.input_size, forwd.shape)
        batch_size = forwd.shape[0]

        """ init hidden and cell state """
        hx = torch.zeros(batch_size, self.hidden_size).cuda()
        cx = torch.zeros(batch_size, self.hidden_size).cuda()
        history = []

        """ forward through lstm """
        for seq in range(forwd.shape[-1]):
            input_data = forwd[..., seq]
            hx, cx = self.lstm_cell(input_data, (hx, cx))
            history.append(hx)

        stacked = torch.stack(history).transpose(0, 1)
        return stacked


class DynamicDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=6,
                 key_size=128, query_size=128, value_size=512,
                 image_size=2051, filter_size=512,
                 num_heads=1,
                 drop_prob=0.5, use_bias=True,
                 filter_activation=nn.Tanh(),
                 policy_activation=nn.Softmax(dim=-1)):
        super(DynamicDecoder, self).__init__()

        """ policy variables """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_prob = drop_prob
        self.use_bias = use_bias
        self.hx = None
        self.cx = None

        """ attention variables """
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size

        """ image feature pre-processing variables """
        self.image_size = image_size
        self.filter_size = filter_size

        """ attention linear layers and activations """
        self.fc_key = nn.Linear(self.value_size, self.key_size, bias=self.use_bias)
        self.fc_query = nn.Linear(self.hidden_size, self.query_size, bias=self.use_bias)
        self.softmax = nn.Softmax(dim=1)
        self.filter_activation = filter_activation
        self.num_heads = num_heads
        self.heads = [nn.Linear(
            self.value_size, self.filter_size
                ).to(device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                     ) for _ in range(self.num_heads)]

        """ policy layers and activation"""
        self.bottleneck = nn.Conv1d(self.image_size, self.filter_size, 1, stride=1, padding=0, bias=self.use_bias)
        self.fc_action = nn.Linear(7, 7, bias=True)
        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias=self.use_bias)
        self.linear = nn.Linear(self.hidden_size, self.output_size, bias=self.use_bias)
        self.drop = nn.Dropout(p=self.drop_prob)
        self.drop_h = nn.Dropout(p=0.2)
        self.policy_activation = policy_activation

        """ init LSTM weights"""
        for name, param in self.lstm_cell.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def init_lstm_cell(self, batch_size):
        self.hx = torch.zeros(batch_size, self.hidden_size).cuda()
        self.cx = torch.zeros(batch_size, self.hidden_size).cuda()

    def forward(self, x, value, action, init_lstm_state=True):
        assert x.shape[0] == value.shape[0]
        assert x.shape[0] == action.shape[0]
        batch_size = x.shape[0]

        if init_lstm_state:
            self.init_lstm_cell(batch_size)

        """ value shape: [B, T, 512] -> key shape: [B, T, 128] """
        key = F.relu(self.fc_key(value))

        """ hx shape: [B, 512] -> query shape: [B, 128, 1]"""
        query = F.relu(self.fc_query(self.hx))
        query = query.unsqueeze(dim=-1)

        """ scaled-dot-product attention """
        scale_1 = torch.sqrt(torch.tensor(key.shape[-1], dtype=torch.double))
        scaled_dot_product = torch.bmm(key, query) / scale_1  # shape: [B, T, 1]
        softmax = self.softmax(scaled_dot_product)  # shape: [B, T, 1]
        element_wise_product = value*softmax  # shape: [B, T, 512]
        current_instruction = torch.sum(element_wise_product, dim=1)  # shape: [B, 512]

        """ dynamic convolutional filters """
        dynamic_filter = torch.stack([head(self.drop_h(current_instruction)) for head in self.heads]).transpose(0, 1)
        dynamic_filter = self.filter_activation(dynamic_filter)
        dynamic_filter = F.normalize(dynamic_filter, p=2, dim=-1)

        """ Key must be in the format [Batch, Channels, L]; Channels == image_size """
        if x.shape[1] != self.image_size:
            x = x.transpose(1, 2)

        x = self.bottleneck(x)

        """ [36, N] = T[512, 36] * T[N, 512] """
        scale_2 = torch.sqrt(torch.tensor(x.shape[1], dtype=torch.double))
        attention_map = torch.bmm(x.transpose(1, 2), dynamic_filter.transpose(-1, -2)) / scale_2
        b, c, f = attention_map.shape
        attention_map = attention_map.reshape(b, c*f)

        action_embedded = self.fc_action(action.cuda())
        in_data = torch.cat((attention_map, action_embedded), 1)

        """ Shape of in_data must be [Batch, Input_size] """
        self.hx, self.cx = self.lstm_cell(in_data, (self.hx, self.cx))

        policy_data = self.hx

        drop = self.drop(policy_data)
        pred = self.linear(drop)
        logits = self.policy_activation(pred)

        return pred, logits, attention_map.reshape(b, c, f)
