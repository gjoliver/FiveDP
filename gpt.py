# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false

import logging
import math
from dataclasses import dataclass
import os

from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2Tokenizer


WORLD_SIZE = 8
BATCH_SIZE = 1


LOGGER = None


def _init_logger(rank):
        global LOGGER

        LOGGER = logging.getLogger(f'rank_{rank}')
        LOGGER.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'[Rank {rank}] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


@dataclass
class GPTConfig:
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304

    context_length: int = 1024

    n_layer: int = 5
    n_head: int = 8
    n_embd: int = 256

    dropout: float = 0.0
    bias: bool = True

    dp_size: int = 1  # DDP
    fsdp_size: int = 2  # FSDP
    cp_size: int = 2  # Context Parallel
    tp_sp_size: int = 2  # Tensor and Sequence Parallel


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5,
        )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # embedding size needs to be evenly split to multiple heads.
        assert config.n_embd % config.n_head == 0

        self.n_embd = config.n_embd
        self.d_head = config.n_embd // config.n_head
        self.dropout = config.dropout

        # key, query, value projections for all heads, but in a batch
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask):
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch.
        q, k, v  = self.qkv(x).chunk(chunks=3, dim=2)

        # Mask out padding tokens. Input attn_mask is a DTensor at this point,
        # because of the sharding plan.
        attn_mask = attn_mask.to_local()
        k = k.masked_fill(attn_mask.unsqueeze(-1), 0)
        v = v.masked_fill(attn_mask.unsqueeze(-1), 0)

        # [B, T, nh, dh] -> [B, nh, T, dh]
        # Use -1 to calculate how many attention heads there are per TP rank.
        k = k.view(B, T, -1, self.d_head).transpose(1, 2)
        q = q.view(B, T, -1, self.d_head).transpose(1, 2)
        v = v.view(B, T, -1, self.d_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # Can't use attn_mask here when is_causal=True.
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        # re-assemble all head outputs side by side.
        # [B, nh, T, dh] -> [B, T, D].
        # Agani use -1 to automatically calculate embedding size per TP rank.
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # output projection
        y = self.resid_dropout(self.proj(y))

        return y


class AttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        # Residue Attention. LayerNorm before Attention.
        x = x + self.attn(self.ln_1(x), attn_mask)
        # Residue MLP. LayerNorm before MLP.
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None

        self.config = config

        # Embedding.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_length, config.n_embd)
        # Transformer blocks.
        self.drop = nn.Dropout(config.dropout)
        self.attns = nn.ModuleList([AttnBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        # Logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight typing.
        # https://paperswithcode.com/method/weight-tying
        self.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for n, p in self.named_parameters():
            if n.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        LOGGER.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pos(self, input_ids):
        device = input_ids.device
        _, seq_len = input_ids.size()
        assert seq_len <= self.config.context_length, (
            f"Cannot forward sequence of length {seq_len}, "
            f"block size is only {self.config.context_length}"
        )
        return torch.arange(0, seq_len, dtype=torch.long, device=device)

    def forward(self, input_ids, attn_mask, is_training: bool = False):
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.wte(input_ids)
        # position embeddings of shape (t, n_embd)
        pos_emb = self.wpe(self._pos(input_ids))

        # Random input drop out.
        x = self.drop(tok_emb + pos_emb)
        # Transformer stack.
        for block in self.attns:
            x = block(x, attn_mask)
        # LayerNorm before logits head.
        x = self.ln_f(x)

        if is_training:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
        else:
            # inference-time: only forward the lm_head on the very last position.
            # note: using list [-1] to preserve the sequence dim.
            logits = self.lm_head(x[:, [-1], :])

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_length
            clipped_input_ids = (
                input_ids
                if input_ids.size(1) <= self.config.context_length
                else input_ids[:, -self.config.context_length:]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(clipped_input_ids)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids


def _optimizer(gpt: torch.nn.Module):
    param_dict = {n: p for n, p in gpt.named_parameters() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': 1e-1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(optim_groups, lr=6e-2, betas=(0.9, 0.95))


def _dataloader(replicas: int, dp_rank: int):
    LOGGER.info(f"Creating dataloader, {replicas} replicas, dp_rank {dp_rank}.")

    def _collate(batch):
        return [
            row["title"] + " " + row["story"] for row in batch
        ]

    dataset = load_dataset("FareedKhan/1k_stories_100_genre", split="train")
    sampler = DistributedSampler(
        dataset,
        num_replicas=replicas,
        rank=dp_rank,
        shuffle=False,  # Be consistent across TP/SP ranks.
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE,
        collate_fn=_collate,
        shuffle=False,  # Be consistent across TP/SP ranks.
    )
    return dataloader


def _init_dist(world_size: int, rank: int):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '8888'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # Initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, device_id=torch.device("cuda:0"),
    )


def _apply_hsdp(model, device_mesh) -> torch.nn.Module:
    # Parallelize all transformer blocks.
    for child_module in model.modules():
        if isinstance(child_module, AttnBlock):
            fully_shard(
                child_module, mesh=device_mesh, reshard_after_forward=True,
            )

    # Also shard embedding & logit head at top level.
    fully_shard(model, mesh=device_mesh, reshard_after_forward=True)


def _apply_sp_tp(model, stp_mesh) -> torch.nn.Module:
    # Apply sequence parallel to model level modules.
    # Namely, we want to shard input embeddings (wte and wpe) ColumnWise
    # for input into the transformer blocks.
    # We will then enable SequenceParallel on the last LayerNorm after transformer blocks,
    # and have the logits head gather all output.
    parallelize_module(
        model,
        stp_mesh,
        {
            "wte": RowwiseParallel(
                input_layouts=Replicate(),
                # Desired output is shard on dimension 1 (sequence parallel).
                output_layouts=Shard(1),
            ),
            "wpe": RowwiseParallel(
                input_layouts=Replicate(),
                # Desired output is shard on dimension 0 (sequence parallel).
                # Note that positional embedding is a 1-D array that gets
                # applied to all prompts in a batch. So dimension 0 is
                # the sequence dimension.
                output_layouts=Shard(0),
            ),
            # LayerNorm needs to handle sequence parallel because above.
            "ln_f": SequenceParallel(),
            # Logits head gathers all output for now.
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                # Not sharding model output for now.
                # So output layer is Replicate(), while use_local_output is True.
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.attns:
        layer_plan = {
            "ln_1": SequenceParallel(),
            "attn": PrepareModuleInput(
                # input_ids is sharded in sequence dim because of ln_1.
                # attn_mask is not sharded.
                input_layouts=(Shard(1), Replicate()),
                # Both inputs to attention module should be fully replicated.
                desired_input_layouts=(Replicate(), Replicate()),
            ),
            "attn.qkv": ColwiseParallel(),  # Columnwise QKV projection.
            # Rowwise FFN projection.
            # Output should be sharded on sequence dimension to prepare for
            # the sequence parallelized MLP LayerNorm.
            "attn.proj": RowwiseParallel(output_layouts=Shard(1)),
            "ln_2": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),  # Because of ln_2 SequenceParallel.
                desired_input_layouts=(Replicate(),),  # MLP itself is TP.
            ),
            "mlp.fc": ColwiseParallel(),
            # Output should be sharded on sequence dimension to prepare for
            # the sequence parallelized Attn LayerNorm of the next block.
            "mlp.proj": RowwiseParallel(output_layouts=Shard(1)),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=stp_mesh,
            parallelize_plan=layer_plan,
        )


def _get_dp_rank(device_mesh, global_rank):
    shape = device_mesh.mesh.shape
    # This is the rank we use to partition input data.
    rank_coords = (device_mesh.mesh == global_rank).nonzero().flatten()
    # We assume the outer most dims are "ddp" and "fsdp",
    # so there must be at least 2 indices in the coordinates array.
    assert len(rank_coords) >= 2
    # Global DP rank is the ddp_idx * ddp_size + fsdp_idx.
    return rank_coords[0] * shape[0] + rank_coords[1]


def _checkpoint(config, model, optimizer):
    state_dict = {
        "cfg": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    fs_storage_writer = FileSystemWriter("./dist_checkpoint/")
    torch.distributed.checkpoint.save(
        state_dict=state_dict,
        storage_writer=fs_storage_writer,
    )


def train_loop(rank: int):
    device = torch.device("cuda:0")

    cfg = GPTConfig()

    # 2D device mesh on CPU.
    # Simulate DDP between instances, and FSDP between GPUs on a same instance.
    device_mesh = dist.init_device_mesh(
        device_type="cuda",
        mesh_shape=(cfg.dp_size, cfg.fsdp_size, cfg.cp_size, cfg.tp_sp_size),
        mesh_dim_names=("ddp", "fsdp", "cp", "sp/tp",),
    )

    # Flatten "fsdp" and "cp" into the "fsdp_cp" mesh.
    # This is necessary when both fsdp and cp are enabled.
    device_mesh["fsdp", "cp"]._flatten(mesh_dim_name="fsdp/cp")

    # Prepare the model.
    gpt = GPT(cfg).to(device)
    # SP & TP.
    _apply_sp_tp(gpt, device_mesh["sp/tp"])
    # HSDP: Inter-node DDP + intra-node FSDP.
    _apply_hsdp(gpt, device_mesh["ddp", "fsdp/cp"])

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Optimizer.
    optimizer = _optimizer(gpt)

    # Dataloader.
    dataloader = _dataloader(
        replicas=device_mesh.size(), dp_rank=_get_dp_rank(device_mesh, rank),
    )

    for i, batch in enumerate(dataloader):
        LOGGER.info(f"step {i}")

        # Input and labels.
        inputs = tokenizer(
            batch,
            padding=True,
            pad_to_multiple_of=(cfg.cp_size * cfg.tp_sp_size),
            truncation=True,
            return_tensors='pt',
        )

        input_ids = inputs["input_ids"].to(device)
        # Mask out paddings.
        attn_mask = inputs["attention_mask"].to(torch.bool).to(device)

        context_parallel_ctx = context_parallel(
			mesh=device_mesh["cp"],
			buffers=[input_ids, attn_mask],
            # shard on seq dimension
			buffer_seq_dims=[1, 1],
			no_restore_buffers={input_ids, attn_mask},
		)

        with context_parallel_ctx:
            # Compute probabilities.
            logits = gpt.forward(
                input_ids=input_ids, attn_mask=attn_mask, is_training=True,
            )

            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = input_ids[..., 1:].contiguous()

            # Cross-entropy loss.
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )

            LOGGER.info(f"loss: {loss}")

            # Gradient step.
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    # Checkpoint at the end of training.
    _checkpoint(cfg, gpt, optimizer)


def train(world_size: int, rank: int):
    _init_dist(world_size, rank)
    _init_logger(rank)

    # Autocast because torch flash attention only works with half precision.
    with torch.autocast(device_type="cuda"):
        # As of 2.7.0, context parallel only works with FLASH_ATTENTION kernel.
        # I did run into weird problems like output tensor has a different shape
        # when I tried memory efficient SDPA kernel.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            train_loop(rank)

    dist.destroy_process_group()


def launch():
    mp.set_start_method('spawn')

    processes = []
    for i in range(WORLD_SIZE):
        process = mp.Process(target=train, args=(WORLD_SIZE, i), daemon=True)
        process.start()
        processes.append(process)

    for p in processes:
        p.join()


if __name__ == "__main__":
    launch()
