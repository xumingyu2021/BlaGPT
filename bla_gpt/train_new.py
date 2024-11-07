import datetime
import glob
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_model

# Set environment variables
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"


@dataclass
class Hyperparameters(Coqpit):
    """Training hyperparameters configuration."""

    run_name: str = "nano_gpt+rms_norm+geglu+gqa+softcap"
    compile_model: bool = True
    input_bin: str = "../data/fineweb10B/fineweb_train_*.bin"
    input_val_bin: str = "../data/fineweb10B/fineweb_val_*.bin"
    batch_size: int = 8 * 64
    device_batch_size: int = 32
    sequence_length: int = 1024
    num_iterations: int = 5100
    learning_rate: float = 4e-4
    warmup_iters: int = 250
    warmdown_iters: int = 2000
    weight_decay: float = 0
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 5000
    keep_last_n_checkpoints: int = 5
    save_best_model: bool = True
    precision: str = "bfloat16"


class TeeLogger:
    """Logger that writes to both terminal and file."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


class DataShard:
    """Handles loading and validation of data shards."""

    MAGIC_NUMBER = 20240520
    HEADER_SIZE = 256 * 4
    VERSION = 1

    @staticmethod
    def peek_data_shard(filename: str) -> int:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(DataShard.HEADER_SIZE), dtype=np.int32)

        if header[0] != DataShard.MAGIC_NUMBER:
            print("ERROR: magic number mismatch in the data .bin file!")
            print("---> HINT: Are you passing in a correct file with --input_bin?")
            print(
                "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
            )
            print(
                "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
            )
            exit(1)
        assert header[1] == DataShard.VERSION, "Unsupported version"
        return header[2]

    @staticmethod
    def load_data_shard(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(DataShard.HEADER_SIZE), dtype=np.int32)
            assert header[0] == DataShard.MAGIC_NUMBER, "Magic number mismatch"
            assert header[1] == DataShard.VERSION, "Unsupported version"
            ntok = header[2]
            tokens = np.frombuffer(f.read(), dtype=np.uint16)

        assert len(tokens) == ntok, "Token count mismatch"
        return tokens


class DistributedDataLoader:
    """Distributed data loader for handling multiple data shards."""

    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_length: int,
        process_rank: int,
        num_processes: int,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")

        self.ntok_total = self._validate_shards()
        self.reset()

    def _validate_shards(self) -> int:
        ntok_total = 0
        min_required = self.num_processes * self.batch_size * self.seq_length + 1

        for fname in self.files:
            shard_ntok = DataShard.peek_data_shard(fname)
            assert shard_ntok >= min_required, f"Shard {fname} too small"
            ntok_total += int(shard_ntok)
        return ntok_total

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.batch_size * self.seq_length
        self.tokens = DataShard.load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size * self.seq_length
        self.tokens = DataShard.load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.seq_length
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x.cuda(), y.cuda()


class Trainer:
    """Main trainer class handling the training loop and validation."""

    def __init__(
        self, args: Hyperparameters, model_name: str, config_path: Optional[str] = None
    ):
        self.args = args
        with open(sys.argv[0]) as f:
            self.code = f.read()
        self.setup_distributed()
        self.setup_model(model_name, config_path)
        self.setup_optimization()
        self.setup_data_loaders()
        if self.is_master:
            self.setup_logging()

    def setup_distributed(self):
        """Initialize distributed training setup."""
        assert torch.cuda.is_available()
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=30)
        )
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = f"cuda:{self.local_rank}"
        self.is_master = self.rank == 0
        torch.cuda.set_device(self.device)

    def setup_model(self, model_name: str, config_path: Optional[str]):
        """Initialize model and move to GPU."""
        model_config, model_cls = get_model(model_name)
        self.model_config = model_config()

        if config_path:
            self.model_config.load_json(config_path)

        self.model = model_cls(self.model_config).cuda()

        if self.args.compile_model:
            self.model = torch.compile(self.model)

        self.model = DDP(
            self.model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        self.raw_model = self.model.module
        self.ctx = autocast(
            dtype=torch.bfloat16 if self.args.precision == "bfloat16" else torch.float32
        )

    def setup_optimization(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
            fused=True,
            foreach=False,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.get_lr_schedule()
        )

    def get_lr_schedule(self):
        """Create learning rate schedule function."""

        def schedule(it):
            if it < self.args.warmup_iters:
                return (it + 1) / self.args.warmup_iters
            elif it < self.args.num_iterations - self.args.warmdown_iters:
                return 1.0
            else:
                return (self.args.num_iterations - it) / self.args.warmdown_iters

        return schedule

    def setup_data_loaders(self):
        """Initialize training and validation data loaders."""
        self.train_loader = DistributedDataLoader(
            self.args.input_bin,
            self.args.device_batch_size,
            self.args.sequence_length,
            self.rank,
            self.world_size,
        )

        self.val_loader = DistributedDataLoader(
            self.args.input_val_bin,
            self.args.device_batch_size,
            self.args.sequence_length,
            self.rank,
            self.world_size,
        )

        self.train_accumulation_steps = self.args.batch_size // (
            self.args.device_batch_size * self.world_size
        )

        self.val_steps = self.args.val_tokens // (
            self.args.device_batch_size * self.args.sequence_length * self.world_size
        )

    def setup_logging(self):
        """Initialize logging directory and files."""
        run_num = 0
        self.run_id = f"{self.args.run_name}_{run_num}"
        self.logdir = f"logs/{self.run_id}/"

        while os.path.exists(self.logdir):
            run_num += 1
            self.run_id = f"{self.args.run_name}_{run_num}"
            self.logdir = f"logs/{self.run_id}/"

        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = f"logs/{self.run_id}.txt"

        sys.stdout = TeeLogger(self.logfile)
        self._log_initial_info()

    def _log_initial_info(self):
        """Log initial information about the training run."""
        print("=" * 100)
        print(self.code)
        print("=" * 100)
        print(
            f"Running pytorch {torch.__version__} compiled for CUDA {torch.version.cuda}"
        )
        print("nvidia-smi:")

        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"{result.stdout}")
        print("=" * 100)

    def validate(self) -> float:
        """Run validation loop and return validation loss."""
        self.model.eval()
        self.val_loader.reset()
        val_loss = 0.0

        with torch.no_grad():
            for _ in range(self.val_steps):
                x_val, y_val = self.val_loader.next_batch()
                with self.ctx:
                    _, loss = self.model(x_val, y_val)

                    if isinstance(loss, dict):
                        loss = loss["total"]
                    val_loss += loss.detach()

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        return val_loss / self.val_steps

    def save_checkpoint(self, step: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        log = {
            "step": step,
            "code": self.code,
            "model_config": self.model_config.to_dict(),
            "train_config": self.args.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if val_loss is not None:
            log["val_loss"] = val_loss

        checkpoint_path = f"{self.logdir}/state_step{step:06d}.pt"
        torch.save(log, checkpoint_path)

        # Cleanup old checkpoints
        if self.args.keep_last_n_checkpoints > 0:
            checkpoints = sorted(glob.glob(f"{self.logdir}/state_step*.pt"))
            if len(checkpoints) > self.args.keep_last_n_checkpoints:
                for checkpoint in checkpoints[: -self.args.keep_last_n_checkpoints]:
                    os.remove(checkpoint)

    def save_best_model(self, step: int, val_loss: float):
        """Save model if it has the best validation loss so far."""
        log = {
            "step": step,
            "code": self.code,
            "model_config": self.model_config.to_dict(),
            "train_config": self.args.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        best_model_path = f"{self.logdir}/best_model_{step}.pt"
        torch.save(log, best_model_path)

        # Remove previous best model if exists
        for f in glob.glob(f"{self.logdir}/best_model_*.pt"):
            if f != best_model_path:
                os.remove(f)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[Dict]]:
        """Execute a single training step."""
        self.model.train()
        metrics = None

        for i in range(1, self.train_accumulation_steps + 1):
            with self.ctx:
                _, loss = self.model(x, y)
                if isinstance(loss, dict):
                    metrics = {k: v for k, v in loss.items() if k != "total"}
                    loss = loss["total"]
                train_loss = loss.detach()

            x, y = self.train_loader.next_batch()

            if i < self.train_accumulation_steps:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= self.train_accumulation_steps

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad(set_to_none=True)

        return train_loss, metrics

    def train(self):
        """Main training loop."""
        training_time_ms = 0
        best_val_loss = float("inf")
        torch.cuda.synchronize()
        t0 = time.time()

        self.train_loader.reset()
        x, y = self.train_loader.next_batch()

        for step in range(self.args.num_iterations + 1):
            last_step = step == self.args.num_iterations

            # Reset timing after first 10 steps
            if step == 10:
                training_time_ms = 0
                t0 = time.time()

            # Calculate timed steps, accounting for warm-up period
            timed_steps = max(1, step - 10) if step > 10 else 1

            # Validation step
            if last_step or (
                self.args.val_loss_every > 0 and step % self.args.val_loss_every == 0
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)

                val_loss = self.validate()

                if self.is_master:
                    self._log_validation_results(
                        step, val_loss, training_time_ms, timed_steps
                    )
                    if self.args.save_best_model and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_best_model(step, val_loss)

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                t0 = time.time()

            # Checkpoint saving
            if self.is_master and (
                last_step
                or (self.args.save_every > 0 and step % self.args.save_every == 0)
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                self.save_checkpoint(step)
                torch.cuda.synchronize()
                t0 = time.time()

            if last_step:
                break

            # Training step
            train_loss, metrics = self.train_step(x, y)

            if self.is_master:
                self._log_training_progress(
                    step, train_loss, training_time_ms, t0, timed_steps, metrics
                )

    def _log_validation_results(
        self, step: int, val_loss: float, training_time_ms: float, timed_steps: float
    ):
        """Log validation results."""
        log_msg = f"step:{step}/{self.args.num_iterations} val_loss:{val_loss:.4f} "
        log_msg += f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms"
        print(log_msg)
        with open(self.logfile, "a") as f:
            f.write(log_msg + "\n")

    def _log_training_progress(
        self,
        step: int,
        train_loss: torch.Tensor,
        training_time_ms: float,
        t0: float,
        timed_steps: float,
        metrics: Optional[Dict] = None,
    ):
        """Log training progress."""
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        lr = self.optimizer.param_groups[0]["lr"]

        log_msg = f"step:{step+1}/{self.args.num_iterations} lr:{lr} train_loss:{train_loss.item():.4f} "
        log_msg += (
            f"train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
        )

        if metrics:
            log_msg += f" {metrics}"
        print(log_msg)


def main():
    """Entry point for training."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bla_gpt")
    args = parser.parse_args()

    hyperparams = Hyperparameters()
    if args.run_name:
        hyperparams.run_name = args.run_name

    trainer = Trainer(hyperparams, args.model_name, args.config)
    trainer.train()


if __name__ == "__main__":
    main()
