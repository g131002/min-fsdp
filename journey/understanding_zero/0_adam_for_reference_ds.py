from basic import DummyModel
from torch.optim import Adam 
import torch
import deepspeed
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    model = DummyModel()

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                        model=model,
                                                        model_parameters=model.parameters())
    
    deepspeed.init_distributed()

    optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8)
    print("Initial model parameters:", list(model.parameters()))
    input = torch.randn(10, generator=torch.Generator().manual_seed(42))
    target = torch.randn(1, generator=torch.Generator().manual_seed(42))
    output = model(input)

    loss = torch.nn.functional.mse_loss(output, target)

    loss.backward()

    optimizer.step()

    print("Updated model parameters:", list(model.parameters()))

    # write down
    # torch.save(model.state_dict(), "model_ref.pth")


