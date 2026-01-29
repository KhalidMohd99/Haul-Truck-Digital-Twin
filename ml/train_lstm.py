import torch, torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2,64,batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1])
