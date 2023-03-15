import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

indices = torch.tensor(dataset.UserItemNet.indices)
indptr = torch.tensor(dataset.UserItemNet.indptr)
data = torch.tensor(dataset.UserItemNet.data)
print(torch.sparse.sum(torch.sparse_csr_tensor(indptr, indices, data), 0))
# print(torch.sparse_csr_tensor(indptr, indices, data, dtype=torch.float64).t())
