import numpy as np
import torch
import torch_ext_test
import torch.nn.functional as F
# import time

x_o = np.random.randint(-126, 126, (1, 900, 512))
# x_o = np.random.randn(1, 900, 512)
w_o = np.ones((512, 512))
b_o = np.ones((512))
# b_o = np.random.randn(512)

x_8 = torch.tensor(x_o, dtype=torch.float).cuda()
w_8 = torch.tensor(w_o, dtype=torch.float).cuda()
b_8 = torch.tensor(b_o, dtype=torch.float).cuda()

# x = torch.tensor(x_o, dtype=torch.float).cuda()
# w = torch.tensor(w_o, dtype=torch.float).cuda()
# b = torch.tensor(b_o, dtype=torch.float).cuda()


for _ in range(1234):
    out_my = torch_ext_test.my_func_mm_bc(x_8, w_8, b_8)



