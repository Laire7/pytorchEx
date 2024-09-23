import torch
cpu = torch.FloatTensor([1,2,3])
gpu = cpu.cuda()
gpu2cpu = gpu.cpu()
cpu2gpu = cpu.to('cuda')
print(cpu)
print(gpu)
print(gpu2cpu)
print(cpu2gpu)