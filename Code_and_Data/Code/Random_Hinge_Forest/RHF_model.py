import torch
from HingeTree import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"
x = torch.rand([100, 1000]).to(device)
timings = HingeTree.speedtest(x)