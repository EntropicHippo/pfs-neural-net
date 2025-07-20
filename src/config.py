import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# === PATHS === #
datafile = '../params/increasing.txt'
checkpoint_path = '../models/model_gnn_dev.pth'
model_pre_trained = '../models/model_gnn_heavy.pth'
# model_pre_trained = ''

# === CONSTANTS === #
NFIBERS = 2000  # number of fibers
NCLASSES = 12   # number of classes
NFIELDS = 10    # number of GE fields
TOTAL_TIME = 42 # per‑fiber time budget

# === TRAINING PARAMETERS === #
retrain = False
nepochs = 200_000
Fdim = 10 # lifted dimension
lr = 5e-4 # learning rate
pclass = 0.1 # penalty coefficient for class over-utilization
pfiber = 0.1 # penalty coefficient for fiber over-utilization
wutils = 2000.0
wvar = 1.0
sharps = [0.0, 100.0]
min_sharp = 50.0
