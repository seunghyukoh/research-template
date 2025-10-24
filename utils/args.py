from dataclasses import dataclass, field
from typing import List, Literal, Optional


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
