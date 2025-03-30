import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler

from .data import TTSCollator, TTSDataset
from .tokenizer import DacTokenizer, MisakiTokenizer, create_dac_tokenizer_model
from .transformer import TTSTransformer, TTSTransformerArgs

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "training_loss.log")
formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_learning_rate: float = 9e-4
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    optimizer: str = "AdamW"
    lr_scheduler: str = "OneCycleLR"

    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    text_pad_id: int = 0
    audio_pad_id: int = 0

    num_workers: int = 48
    seed: int = 42
    backend: str = "nccl"

    checkpoint_epoch_freq: int = 1
    validation_freq: int = 1
    checkpoint_steps_freq: int = 2000
    global_step_log_freq: int = 10
    validation_epoch_logger_freq: int = 100
    resume_checkpoint: Optional[Path] = None

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")


@dataclass
class CheckpointState:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    trainer_state: TrainerState
    config: TrainingConfig
    scaler_state: Optional[Dict[str, Any]] = None

    def save(self, path: Path):
        torch.save(
            {
                "model_state": self.model_state,
                "optimizer_state": self.optimizer_state,
                "scheduler_state": self.scheduler_state,
                "trainer_state": self.trainer_state,
                "config": self.config,
                "scaler_state": self.scaler_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "CheckpointState":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return cls(
            model_state=checkpoint["model_state"],
            optimizer_state=checkpoint["optimizer_state"],
            scheduler_state=checkpoint.get("scheduler_state"),
            trainer_state=checkpoint["trainer_state"],
            config=checkpoint["config"],
            scaler_state=checkpoint["scaler_state"],
        )


class DistributedTTSTrainer:
    def __init__(
        self, config: TrainingConfig, model: torch.nn.Module, device: torch.device
    ):
        self.config: TrainingConfig = config
        self.device = device
        self.is_main = dist.get_rank() == 0

        self.model = DDP(
            model.to(device), device_ids=[device.index], output_device=device.index
        )
        self.optimizer: Optimizer = self._create_optimizer()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.scheduler = self._create_scheduler()

        self.state = TrainerState()
        self.scaler = GradScaler()
        if self.config.resume_checkpoint:
            self.load_checkpoint()

    def _create_optimizer(self):
        if self.config.optimizer == "AdamW":
            return AdamW(self.model.parameters(), lr=self.config.learning_rate)
        else:
            return NotImplementedError(
                f"Optimizer: {self.config.optimizer} not implemented!"
            )

    def _create_scheduler(self):
        if self.config.lr_scheduler == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_learning_rate,
                total_steps=self.config.epochs
                * (
                    (len(self.train_loader) + self.config.grad_accum_steps - 1)
                    // self.config.grad_accum_steps
                ),
            )
        else:
            return NotImplementedError(
                f"lr_scheduler:{self.config.lr_scheduler} not implemented!"
            )

    def _create_dataloaders(self):
        collator_fn = TTSCollator(self.config.text_pad_id, self.config.audio_pad_id)

        train_dataset = TTSDataset("train", self.config.data_dir)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            collate_fn=collator_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val_dataset = TTSDataset("validation", self.config.data_dir)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            collate_fn=collator_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    def save_checkpoint(self, best: bool = False):
        if not self.is_main:
            return

        checkpoint = CheckpointState(
            model_state=self.model.module.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            trainer_state=self.state,
            config=self.config,
            scaler_state=self.scaler.state_dict() if self.scaler else None,
        )

        if best:
            filename = "best.pt"
        elif self.state.global_step % self.config.checkpoint_steps_freq == 0:
            filename = f"checkpoint_step_{self.state.global_step:06d}.pt"
        else:
            filename = f"checkpoint_epoch_{self.state.epoch:04d}.pt"
        path = self.config.checkpoint_dir / filename
        checkpoint.save(path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self):
        checkpoint = CheckpointState.load(self.config.resume_checkpoint, self.device)

        self.model.module.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)

        if self.scheduler and checkpoint.scheduler_state:
            self.scheduler.load_state_dict(checkpoint.scheduler_state)

        if checkpoint.scaler_state and self.scaler:
            self.scaler.load_state_dict(checkpoint.scaler_state)

        self.state = checkpoint.trainer_state

        obj_list = [self.state.epoch, self.state.global_step]
        dist.broadcast_object_list(obj_list, src=0)
        self.state.epoch = obj_list[0]
        self.state.global_step = obj_list[1]

        logger.info(
            f"Resumed training from {self.config.resume_checkpoint} at epoch {self.state.epoch}"
        )

    def train_epoch(self):
        self.model.train()
        self.train_loader.sampler.set_epoch(self.state.epoch)
        total_loss = 0.0
        accum_steps = 0

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)
                }

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss, _ = self.model(
                        text_tokens=batch["text_tokens"],
                        audio_tokens=batch["audio_tokens"][:, :, :-1],
                        target=batch["audio_tokens"][:, :, 1:],
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        f"NaN/Inf loss detected at batch {batch_idx}, skipping batch."
                    )
                    continue

                scaled_loss = loss / self.config.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()
                total_loss += loss.item()
                accum_steps += 1

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    total_loss_tensor = torch.tensor(total_loss, device=self.device)
                    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

                    if self.is_main and (
                        self.state.global_step % self.config.global_step_log_freq == 0
                    ):
                        avg_loss = total_loss_tensor.item() / (
                            accum_steps * dist.get_world_size()
                        )
                        logger.info(
                            f"Epoch {self.state.epoch} Step {self.state.global_step} Batch {batch_idx + 1} Loss: {avg_loss:.4f}"
                        )

                    total_loss = 0.0
                    accum_steps = 0

                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.state.global_step += 1

                    if self.state.global_step % self.config.checkpoint_steps_freq == 0:
                        self.save_checkpoint()
            except Exception as e:
                logger.error(
                    f"Training: Skipping batch {batch_idx} due to error: {e}",
                    exc_info=True,
                )
                logger.error(f"Batch items: {batch.items}")
                continue

        self.state.epoch += 1

    def validate(self):
        if not self.val_loader:
            return

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    batch = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if torch.is_tensor(v)
                    }

                    loss, _ = self.model(
                        text_tokens=batch["text_tokens"],
                        audio_tokens=batch["audio_tokens"][:, :, :-1],
                        target=batch["audio_tokens"][:, :, 1:],
                    )

                    num_samples = batch["text_tokens"].size(0)
                    batch_loss = loss.item() * num_samples
                    total_loss += batch_loss
                    total_samples += num_samples

                    if (
                        batch_idx + 1
                    ) % self.config.validation_epoch_logger_freq == 0 and self.is_main:
                        avg_loss_so_far = (
                            total_loss / total_samples if total_samples != 0 else 0.0
                        )
                        logger.info(
                            f"Validation: Batch {batch_idx + 1} Loss: {avg_loss_so_far:.4f}"
                        )

                except Exception as e:
                    logger.error(
                        f"Validation: Skipping batch {batch_idx} due to error: {e}",
                        exc_info=True,
                    )
                    continue

        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        total_samples_tensor = torch.tensor(total_samples, device=self.device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        avg_loss = (
            total_loss_tensor.item() / total_samples_tensor.item()
            if total_samples_tensor.item() != 0
            else 0.0
        )

        if self.is_main:
            logger.info(f"Validation Loss: {avg_loss:.4f}")
            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self.save_checkpoint(best=True)


def setup_distributed(config: TrainingConfig) -> torch.device:
    dist.init_process_group(backend=config.backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(config.seed + dist.get_rank())
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Process {os.getpid()} using GPU: {device}")
    return device


def train(config: TrainingConfig):
    device = setup_distributed(config)

    dac_model = create_dac_tokenizer_model("16khz")
    dac_tokenizer = DacTokenizer(dac_model)
    misaki_tokenizer = MisakiTokenizer()

    ttsTransformerArgs = TTSTransformerArgs(
        text_vocab_size=misaki_tokenizer.vocab_size,
        text_pad_id=config.text_pad_id,
        audio_pad_id=config.audio_pad_id,
        audio_vocab_size=dac_tokenizer.vocab_size,
    )
    model = TTSTransformer(ttsTransformerArgs)

    trainer = DistributedTTSTrainer(config, model, device)

    start_epoch = trainer.state.epoch
    for epoch in range(start_epoch, config.epochs):
        trainer.train_epoch()

        if (epoch + 1) % config.validation_freq == 0:
            trainer.validate()

        if (epoch + 1) % config.checkpoint_epoch_freq == 0:
            trainer.save_checkpoint()

    dist.destroy_process_group()


def validate(config: TrainingConfig):
    device = setup_distributed(config)

    dac_model = create_dac_tokenizer_model("16khz")
    dac_tokenizer = DacTokenizer(dac_model)
    misaki_tokenizer = MisakiTokenizer()

    ttsTransformerArgs = TTSTransformerArgs(
        text_vocab_size=misaki_tokenizer.vocab_size,
        text_pad_id=config.text_pad_id,
        audio_pad_id=config.audio_pad_id,
        audio_vocab_size=dac_tokenizer.vocab_size,
    )
    model = TTSTransformer(ttsTransformerArgs)

    trainer = DistributedTTSTrainer(config, model, device)

    trainer.validate()

    dist.destroy_process_group()


if __name__ == "__main__":
    # config = TrainingConfig(resume_checkpoint=Path("checkpoints/checkpoint_0001.pt"))
    # config = TrainingConfig(resume_checkpoint=Path("checkpoints/best.pt"))
    config = TrainingConfig(
        resume_checkpoint=Path("checkpoints/checkpoint_epoch_0016.pt")
    )
    # config = TrainingConfig(resume_checkpoint=None)
    train(config)
    # validate(config)
