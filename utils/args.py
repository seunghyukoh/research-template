from dataclasses import dataclass, field
from typing import List, Literal, Optional

from trl import SFTConfig as TRLSFTConfig


@dataclass
class SFTConfig(TRLSFTConfig):
    auto_batch_size: Optional[bool] = field(
        default=False,
        metadata={"help": "Automatically set the largest possible batch size."},
    )


@dataclass
class WandbConfig:
    entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity (team) name"})
    project: Optional[str] = field(default=None, metadata={"help": "Wandb project name"})
    id: Optional[str] = field(default=None, metadata={"help": "Run ID for wandb"})
    notes: Optional[str] = field(default=None, metadata={"help": "Notes for wandb run"})
    tags: List[str] = field(default_factory=list, metadata={"help": "Tags for wandb run"})
    group: Optional[str] = field(default=None, metadata={"help": "Group name for wandb run"})
    job_type: Optional[str] = field(default=None, metadata={"help": "Job type for wandb run"})
    mode: Optional[str] = field(
        default=None, metadata={"help": "Wandb mode: online, offline, disabled"}
    )
    resume: Optional[Literal["allow", "never", "must", "auto"]] = field(
        default=None,
        metadata={"help": "Resume behavior for wandb"},
    )
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Run ID to resume from"},
    )
    save_code: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save the code to wandb"},
    )

    def __post_init__(self):
        from datetime import datetime, timezone

        if self.id is None:
            from uuid import uuid4

            self.id = uuid4().hex[:8]

        self.timestamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M")
