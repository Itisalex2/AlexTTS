import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from contextlib import ExitStack


from lingua.tokenizer import build_tokenizer
from lingua.checkpoint import CheckpointArgs
from lingua.data import (
    DataArgs,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
)
from lingua.metrics import (
    LoggingArgs,
)
from lingua.optim import OptimArgs
from lingua.profiling import ProfilerArgs
from apps.main.transformer import (
    LMTransformerArgs,
)


logger = logging.getLogger()


@dataclass
class TrainArgs:
    name: str = "text-to-speech"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


def train(args: TrainArgs):
    with ExitStack():
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        logger.info(f"Tokenizer: {tokenizer}")
