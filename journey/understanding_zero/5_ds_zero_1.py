import copy

import argparse
import deepspeed
import torch

from basic import DummyModel, check_model_from_reference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    model = DummyModel()

    copy_model = copy.deepcopy(model)

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                        model=model,
                                                        model_parameters=model.parameters())
    
    input = torch.randn(10, generator=torch.Generator().manual_seed(42)).to(model_engine.device)
    target = torch.randn(1, generator=torch.Generator().manual_seed(42)).to(model_engine.device)

    #forward() method
    # with amp.autocast():
    output = model_engine(input)
    loss = torch.nn.functional.mse_loss(output, target)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

    check_model_from_reference(model_engine.module, "model_ref_ds.pth")

    print('test')
