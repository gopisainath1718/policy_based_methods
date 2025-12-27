import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, s_size, h_size, a_size):
        super().__init__()

        self.fc1 = nn.
