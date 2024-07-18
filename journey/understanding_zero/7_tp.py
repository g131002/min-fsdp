import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.autograd import Function

from basic import DummyModel, check_model_from_reference

class AllGather(Function):
    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        gathered_inputs = [torch.zeros_like(input) for _ in range(world_size)]
        dist.all_gather(gathered_inputs, input)
        return torch.cat(gathered_inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        local_grad_size = grad_output.shape[0] // world_size
        local_grad = grad_output[dist.get_rank() * local_grad_size: (dist.get_rank() + 1) * local_grad_size]
        return local_grad


# class ColParallelLinear(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ColParallelLinear, self).__init__()
#         assert output_size % dist.get_world_size() == 0
#         self.local_output_size = output_size // dist.get_world_size()
#         self.weight = nn.Parameter(torch.randn(self.local_output_size, input_size))
#         self.bias = nn.Parameter(torch.randn(self.local_output_size))

#     def forward(self, x):
#         output = torch.matmul(x, self.weight.t()) + self.bias
#         gathered_output = [torch.zeros_like(output)] * dist.get_world_size()
#         dist.all_gather(gathered_output, output)
#         return torch.cat(gathered_output, dim=-1)

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = ColParallelLinear(10, 10)
#         self.fc2 = torch.nn.Linear(10, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
class Model(nn.Module):
    def __init__(self, dummy_model):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        with torch.no_grad():
            if dist.get_rank() == 0:
                self.fc1.weight.copy_(dummy_model.fc1.weight[:5, :])
                self.fc1.bias.copy_(dummy_model.fc1.bias[:5])
            else:
                self.fc1.weight.copy_(dummy_model.fc1.weight[5:, :])
                self.fc1.bias.copy_(dummy_model.fc1.bias[5:])
        self.fc2 = torch.nn.Linear(10, 1)
        with torch.no_grad():
            self.fc2.weight.copy_(dummy_model.fc2.weight)
            self.fc2.bias.copy_(dummy_model.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        all_gather = AllGather.apply
        x = all_gather(x)
        # gathered_output = [torch.zeros_like(x)] * dist.get_world_size()
        # dist.all_gather(gathered_output, x)
        # x = torch.cat(gathered_output, dim=-1)
        x = self.fc2(x)
        return x

def train_test():
    dist.init_process_group(backend="nccl", init_method="env://")
    dummy_model = DummyModel()

    input = torch.randn(10, generator=torch.Generator().manual_seed(42))
    target = torch.randn(1, generator=torch.Generator().manual_seed(42))

    # output = dummy_model(input)
    # loss = torch.nn.functional.mse_loss(output, target)
    # loss.backward()
    rank = int(os.environ["LOCAL_RANK"])

    device = f"cuda:{rank}"

    input = input.to(device)
    target = target.to(device)

    model = Model(dummy_model).to(device)

    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8)
    optimizer.step()

    gathered_fc1_weight = [torch.zeros_like(model.fc1.weight) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_fc1_weight, model.fc1.weight)
    gathered_fc1_weight = torch.cat(gathered_fc1_weight)
    gathered_fc1_bias = [torch.zeros_like(model.fc1.bias) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_fc1_bias, model.fc1.bias)
    gathered_fc1_bias = torch.cat(gathered_fc1_bias)

    with torch.no_grad():
        dummy_model.fc1.weight.copy_(gathered_fc1_weight)
        dummy_model.fc1.bias.copy_(gathered_fc1_bias)
        dummy_model.fc2.weight.copy_(model.fc2.weight)
        dummy_model.fc2.bias.copy_(model.fc2.bias)

    check_model_from_reference(dummy_model)  # Nice!

    print('test')
    
if __name__ == "__main__":
    train_test()
