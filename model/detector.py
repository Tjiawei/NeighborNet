import torch
import torch.nn as nn
import torch.nn.functional as F


class Shotcol(nn.Module):
    def __init__(self, in_ch=2048, hid_ch=2048, out_ch=1, win_size=3):
        super(Shotcol, self).__init__()
        self.linear_1 = nn.Linear(in_ch*win_size*2, hid_ch)
        self.linear_2 = nn.Linear(hid_ch, hid_ch//2)
        self.linear_3 = nn.Linear(hid_ch//2, out_ch)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.act = nn.Sigmoid()

        self.win = win_size

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)
        self.initialize_weight(self.linear_3)

    def forward(self, x):
        batch = x.shape[0]
        half = x.shape[1]//2
        _, x, _ = torch.split(x, [half-self.win, 2*self.win, half-self.win], dim=1)
        x = x.view(batch, -1)
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.relu(self.linear_2(x))
        x = self.drop(x)
        x = self.act(self.linear_3(x))

        return x.squeeze()

    def analyze(self, x):
        batch = x.shape[0]
        half = x.shape[1] // 2
        _, x, _ = torch.split(x, [half - self.win, 2 * self.win, half - self.win], dim=1)
        x = x.view(batch, -1)
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.relu(self.linear_2(x))
        x = self.drop(x)
        x = self.act(self.linear_3(x))

        return x.squeeze()

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, drop=0.5):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(in_ch, hid_ch)
        self.linear_2 = nn.Linear(hid_ch, out_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)

    def forward(self, x):
        """
        :param x: (B, T, C)
        :return:
        """
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.linear_2(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class ProcessTS(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        self.conv1_1 = nn.Conv1d(in_channel, 64, 3, padding=1, padding_mode='replicate', bias=True)
        self.conv1_2 = nn.Conv1d(64, 64, 3, padding=1, padding_mode='replicate', bias=True)
        self.mpool = nn.MaxPool1d(2, stride=2)
        self.conv2_1 = nn.Conv1d(64, 128, 3, padding=1, padding_mode='replicate', bias=True)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=1, padding_mode='replicate', bias=True)
        self.fconv = nn.Conv1d(128, 1, 1, padding=0, bias=True)
        self.relu = nn.ReLU()

        self.initialize_weight(self.conv1_1)
        self.initialize_weight(self.conv1_2)
        self.initialize_weight(self.conv2_1)
        self.initialize_weight(self.conv2_2)
        self.initialize_weight(self.fconv)

    def forward(self, x):
        """
        :param x: (batch, T)
        :return:
            x:(batch, T/2, T/2)
        """
        x = x[:, None]
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.mpool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.fconv(x)
        x = torch.clamp(x, -1.0, 1.0)

        x = torch.squeeze(x, dim=1)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class BaSSLDet(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, drop=0.5):
        super().__init__()
        self.detect = MLP(in_ch, hid_ch, out_ch, drop)

    def forward(self, x):
        out = F.sigmoid(self.detect(x))
        return out


class LatentDetector(nn.Module):
    def __init__(self, seg_sz=20):
        super().__init__()
        self.drop_t = nn.Dropout(0.2)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.tconv = ProcessTS()

        # MLP
        self.cls = MLP(seg_sz//2, 128, 1, 0.2)

    def forward(self, x):
        B, T, _ = x.shape
        half = x.shape[1]//2
        x = F.normalize(x, dim=-1)

        # #  prediction for the center shot
        pre, aft = torch.split(x, half, dim=1)
        pre_t = torch.matmul(aft[:, 0][:, None], pre.transpose(-2, -1))
        aft_t = torch.matmul(pre[:, -1][:, None], aft.transpose(-2, -1))
        sim = torch.concat((pre_t.squeeze(), aft_t.squeeze()), dim=-1)
        sim = self.tconv(sim)
        out = self.act(self.cls(sim))

        # # # prediction for all shots
        # sim = torch.matmul(x, x.transpose(-2, -1))
        # for i in range(1,T-1):
        #     sim[:, i, :i+1] = torch.clone(sim[:, i+1, :i+1])
        #
        # sim = sim.view(B*T, -1).contiguous()
        # sim = self.tconv(sim)
        # out = self.act(self.cls(sim))
        # out = out.view(B, -1)

        return out.squeeze()

    def analyze(self, x):
        B, T, _ = x.shape
        half = x.shape[1] // 2
        x = F.normalize(x, dim=-1)

        # #  prediction for the center shot
        pre, aft = torch.split(x, half, dim=1)
        pre_t = torch.matmul(aft[:, 0][:, None], pre.transpose(-2, -1))
        aft_t = torch.matmul(pre[:, -1][:, None], aft.transpose(-2, -1))
        sim = torch.concat((pre_t.squeeze(dim=1), aft_t.squeeze(dim=1)), dim=-1)
        sim = self.tconv(sim)
        out = self.act(self.cls(sim))

        return out

    def initialize_module(self, modules:list):
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
