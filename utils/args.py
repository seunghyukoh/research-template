from dataclasses import dataclass, field
from typing import List, Literal, Optional

from trl import SFTConfig as TRLSFTConfig


@dataclass
class SFTConfig(TRLSFTConfig):
    auto_batch_size: Optional[bool] = field(
        default=False,
        metadata={"help": "Automatically set the largest possible batch size."},
    )

    def prepare_for_run(self, uid, timestamp, model_name_or_path=None, tags=None):
        self._update_run_name(uid)
        self._update_output_dir(uid, model_name_or_path, tags)
        self._update_revision(timestamp, uid)

    def _update_run_name(self, uid):
        self.run_name += f"-{uid}"

    def _update_output_dir(
        self,
        uid,
        model_name_or_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        assert self.output_dir is not None, "output_dir must be specified"
        from pathlib import Path

        # Output dir = output_dir/model_name_or_path/tags/uid
        parts = [self.output_dir]
        if model_name_or_path:
            model_part = model_name_or_path.replace("/", "_").replace("\\", "_")
            parts.append(model_part)
        if tags:
            tags_part = "_".join(tags)
            parts.append(tags_part)
        parts.append(uid)
        self.output_dir = str(Path(*parts))

    def _update_revision(self, timestamp, uid):
        if not self.push_to_hub:
            return

        from huggingface_hub import HfApi

        api = HfApi()

        # Ensure the repository exists before creating a branch
        api.create_repo(
            repo_id=self.hub_model_id,
            exist_ok=True,
        )

        # Create a unique branch name
        if self.hub_revision:
            self.hub_revision += f"{timestamp}-{uid}"
        else:
            self.hub_revision = f"{timestamp}-{uid}"

        # Create the branch in the repository
        api.create_branch(
            repo_id=self.hub_model_id,
            branch=self.hub_revision,
            exist_ok=True,
        )


@dataclass
class WandbConfig:
    entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity (team) name"})
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
