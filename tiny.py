"""
This file tests 5D parallelism on an extremely simple tiny omdel.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
import torch.multiprocessing as mp
import torch.nn as nn


WORLD_SIZE = 8


class TestModel(nn.Module):
    def __init__(self, rank):
        super().__init__()

        self.rank = rank

        self.wte = nn.Embedding(100, 256)
        self.fc = nn.Linear(256, 64, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.rank == 0:
            print("model.fc in forward:")
            print(f"\tlocal shape:", self.fc.weight._local_tensor.shape)
            print(
                f"\tdevice mesh:",
                self.fc.weight.device_mesh,
                self.fc.weight.device_mesh.get_coordinate(),
                self.fc.weight.placements,
            )

        x = self.wte(x)
        x = self.fc(x)
        x = self.gelu(x)
        return x

    def _init_weights(self):
        torch.nn.init.normal_(self.fc.weight, mean=0.0, std=1.0)


def test(world_size, rank):
    torch.manual_seed(42)

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '8888'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    _ = dist.init_process_group(
        "gloo", rank=rank, world_size=world_size,
    )

    device_mesh = dist.init_device_mesh(
        device_type="cpu",
        mesh_shape=(2, 2, 2),
    	mesh_dim_names=("dp", "fsdp", "tp"),
    )

    model = TestModel(rank)
    model._init_weights()

    # Apply TP.
    parallelize_module(
        model,
        device_mesh["tp"],
        {
            "wte": ColwiseParallel(
                input_layouts=Replicate(),
                # Desired output is shard on dimension 1 (sequence parallel).
                output_layouts=Replicate(),
            ),
            "fc": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # Apply FSDP2.
    fully_shard(model, mesh=device_mesh["dp", "fsdp"], reshard_after_forward=True)

    if rank == 0:
        print("model.fc:")
        print(f"\tlocal shape:", model.fc.weight._local_tensor.shape)
        print(
            f"\tdevice mesh:",
            model.fc.weight.device_mesh,
            model.fc.weight.device_mesh.get_coordinate(),
            model.fc.weight.placements,
        )

    y = model(torch.randint(low=0, high=100, size=(2, 25)))

    print(f"rank {rank}", y.shape)

    dist.destroy_process_group()


def launch():
    mp.set_start_method('spawn')

    processes = []
    for i in range(WORLD_SIZE):
        process = mp.Process(target=test, args=(WORLD_SIZE, i), daemon=True)
        process.start()
        processes.append(process)

    for p in processes:
        p.join()


if __name__ == "__main__":
    launch()
