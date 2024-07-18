import os

import torch
import torch.distributed as dist

from basic import DummyModel, check_model_from_reference

def align_dense_tensors(tensor_list, alignment):
    num_elements = sum(t.numel() for t in tensor_list)
    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list

    return padded_tensor_list

class Zero1AdamOptimizer:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        skip_small_parameters=5,
        forward_dtype=torch.bfloat16,
        fp16_master_weights_and_grads=False
    ):
        # self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.skip_small_parameters = skip_small_parameters

        self.t = 0
        self.partition_id = dist.get_rank()
        self.world_size = dist.get_world_size()
        # self.local_world_size = self.world_size
        # self.device = f"cuda:{self.local_rank}"
        self.offsets = []
        self.shard_indices = []
        self.local_param_indices = set()

        # align nccl all-gather send buffers to 4-byte boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        self.trainable_params = []
        for param_idx, param in enumerate(params):
            if param_idx == 0:
                self.dtype = param.dtype
            if param.requires_grad:
                param.grad_accum = None
                self.trainable_params.append(param)

        orig_numel = 0
        for param in self.trainable_params:
            orig_numel += param.numel()

        self.meta_tensors = []
        for param in self.trainable_params:
            self.meta_tensors.append(torch.zeros_like(param.data, device="meta"))

        self.flattened_buffer = self.flatten_dense_tensors_aligned(
            self.trainable_params, 
            self.nccl_start_alignment_factor * self.world_size
        )

        self.padding = self.flattened_buffer.numel() - orig_numel

        # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
        self.data_parallel_partitions = self.get_data_parallel_partitions(self.flattened_buffer)

        # verify that data partition start locations are 4-byte aligned
        for partitioned_data in self.data_parallel_partitions:
            assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

        if not fp16_master_weights_and_grads:
            self.single_partition_of_fp32 = self.data_parallel_partitions[self.partition_id].clone().float().detach()
        else:
            self.single_partition_of_fp32 = self.data_parallel_partitions[self.partition_id].clone().half().detach()

        self.single_partition_of_fp32.requires_grad = True

        self.partition_size = int(len(self.flattened_buffer) / self.world_size)
        self.v = torch.zeros(self.partition_size).to(self.current_device_name())
        self.m = torch.zeros(self.partition_size).to(self.current_device_name())

        self.params_in_partition, self.params_not_in_partition, self.first_offset = self.get_partition_info(
                self.trainable_params, self.partition_size, self.partition_id)

        self.param_id = {}
        self.param_dict = {}
        largest_param_numel = 0
        self.params_already_reduced = []
        self.is_param_in_current_partition = {}
        count = 0
        for param in self.trainable_params:
            unique_id = id(param)
            self.param_id[unique_id] = count
            self.param_dict[count] = param
            self.params_already_reduced.append(False)
            if param.numel() > largest_param_numel:
                largest_param_numel = param.numel()
            count += 1

        for param in self.params_in_partition:
            self.is_param_in_current_partition[self.get_param_id(param)] = True
        
        for param in self.params_not_in_partition:
            self.is_param_in_current_partition[self.get_param_id(param)] = False

        single_grad_partition = torch.zeros(
            int(self.partition_size),
            dtype=self.single_partition_of_fp32.dtype,
            device=self.current_device_name()
        )

        self.single_partition_of_fp32.grad = single_grad_partition

        self.averaged_gradients = None

        self.step()

    # create a flat tensor aligned at the alignment boundary
    def flatten_dense_tensors_aligned(self, tensor_list, alignment, use_cpu_data=False):
        tensor_list = [param.cpu_data for param in tensor_list] if use_cpu_data else tensor_list
        padded_tensor_list = align_dense_tensors(tensor_list, alignment)
        return torch.cat([torch.flatten(t) for t in padded_tensor_list])
    
    def current_device_name(self):
        return 'cuda:{}'.format(self.partition_id)
    
    # views the tensor as multiple partitions and returns
    # those partitions
    def get_data_parallel_partitions(self, tensor):
        partitions = []

        dp = self.world_size
        # dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions
    
    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if start_index <= current_index < end_index:
                params_in_partition.append(tensor)

            elif current_index < start_index < (current_index + tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset
    
    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def reduce_gradients(self):
        for param in self.trainable_params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self, tensor_list, first_offset, partition_size, dtype, device, return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0

        for i, tensor in enumerate(tensor_list):
            grad_accum = tensor.grad
            if grad_accum is None:
                grad_accum = torch.zeros_like(tensor, dtype=dtype)

            tensor = grad_accum
            num_elements = tensor.numel()
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size

            # we need a narrow view of the tensor based on the tensor offset and number of elements that
            # we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements)))
            else:
                flat_tensor_list.append(tensor)

            current_size = current_size + num_elements

        # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            flat_tensor_list.append(torch.zeros(int(partition_size - current_size), dtype=dtype, device=device))

        if return_tensor_list:
            return flat_tensor_list

        return torch.cat([torch.flatten(t) for t in flat_tensor_list])

    def step(self):
        self.reduce_gradients()

        dist.barrier()

        if not self.averaged_gradients:
            self.averaged_gradients = self.get_flat_partition(
                self.params_in_partition,
                self.first_offset,
                self.partition_size,
                dtype=torch.float32,
                device=self.current_device_name(),
                return_tensor_list=True
            )
        else:
            avg_new = self.get_flat_partition(self.params_in_partition,
                                                self.first_offset,
                                                self.partition_size,
                                                dtype=torch.float32,
                                                device=self.current_device_name(),
                                                return_tensor_list=True)

            for accumulated_grad, new_avg_grad in zip(self.averaged_gradients, avg_new):
                accumulated_grad.add_(new_avg_grad)

        # No need to keep the gradients anymore.
        # All gradients required by the step
        # are in self.averaged_gradients
        self.zero_grad(set_to_none=True)

        # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
        self.free_grad_in_param_list(self.params_not_in_partition)

        # create a flat gradients for parameters updated by this process
        # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
        if self.partition_id == dist.get_world_size() - 1:
            single_grad_partition = self.flatten_dense_tensors_aligned(
                self.averaged_gradients,
                int(self.partition_size)).to(self.single_partition_of_fp32.dtype)
            
        else:
            single_grad_partition = torch.cat([torch.flatten(t) for t in self.averaged_gradients]).to(self.single_partition_of_fp32.dtype)

        assert single_grad_partition.numel() == self.partition_size, \
            "averaged gradients have different number of elements that partition size {} {} {} ".format(
                single_grad_partition.numel(), self.partition_size, self.partition_id)
        
        self.single_partition_of_fp32.grad = single_grad_partition
        # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
        self.free_grad_in_param_list(self.params_in_partition)

        self.averaged_gradients = None

         # These operation happens-per-device!
        self.t += 1.0

        self.adam_step()

        dist.barrier()

        self.single_partition_of_fp32.grad = None
        del single_grad_partition
        bit16_partitions = self.data_parallel_partitions
        fp32_partition = self.single_partition_of_fp32
        bit16_partitions[self.partition_id].data.copy_(fp32_partition.data)

        if self.world_size != 1:
            dist.all_gather_into_tensor(
                self.flattened_buffer,
                self.data_parallel_partitions[self.partition_id]
            )

        updated_params = self.unflatten(self.flattened_buffer, self.padding, self.meta_tensors)
        for p, q in zip(self.trainable_params, updated_params):
            p.data = q.data

    # @torch.compile()
    def adam_step(self):
        # Zero-1 Adam with compile!
        grad = self.single_partition_of_fp32.grad
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        self.m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.single_partition_of_fp32.data -= (
            self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        )

    def zero_grad(self, set_to_none=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        # zero all pointers to grad tensors
        for p in self.trainable_params:
            if set_to_none:
                p.grad = None  # epilogue and in step
                p.grad_accum = None
            else:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None  # in step
            p.grad_accum = None

    def unflatten(self, tensor, padding, meta):
        tensor = tensor[:-padding]
        results = []
        sizes = []
        for param in meta:
            sizes.append(param.shape)
        offset = 0
        for size in sizes:
            size_flattend = 1
            for size_dim in size:
                size_flattend = size_flattend * size_dim
            results.append(torch.unflatten(tensor[offset:offset + size_flattend], 0, size))
            offset += size_flattend
        return results


def train_test():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Running DDP example on rank {rank}, world size: {world_size}")
    dist.init_process_group(backend="nccl", init_method="env://")

    device = f"cuda:{rank}"

    model = DummyModel().to(device)
    forward_dtype = torch.bfloat16
    optimizer = Zero1AdamOptimizer(
        model.parameters(), lr=1e-1, forward_dtype=forward_dtype
    )

    input = torch.randn(10, generator=torch.Generator().manual_seed(42)).to(
        device
    )
    target = torch.randn(1, generator=torch.Generator().manual_seed(42)).to(
        device
    )

    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)

    loss.backward()
    optimizer.step()

    check_model_from_reference(model, "model_ref_ds.pth")  # Nice!

if __name__ == "__main__":
    train_test()