from typing import Any


def resolve_training_timestamp(args: Any, current_timestamp: str) -> str:
    resume_log_timestamp = getattr(args, "resume_log_timestamp", None)
    if resume_log_timestamp is None:
        return current_timestamp
    if getattr(args, "resume_path", None) is None:
        raise ValueError("--resume_log_timestamp requires --resume_path.")
    return resume_log_timestamp
