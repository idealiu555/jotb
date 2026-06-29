from marl_models.base_model import MARLModel
from environment.env import Env
from marl_models.utils import get_model, set_seed
from train import train_on_policy, train_off_policy, train_random
from test import test_model
from utils.logger import Logger
from utils.plot_logs import generate_plots_if_available
from utils.training_run import resolve_training_timestamp
import config
import argparse
import os
import re
from datetime import datetime


def infer_resume_start_step(resume_path: str, progress_name: str) -> int:
    checkpoint_dir: str = os.path.basename(os.path.normpath(resume_path))
    match = re.fullmatch(rf"{re.escape(progress_name)}_(\d+)", checkpoint_dir)
    if match is None:
        raise ValueError(
            f"Resume path '{resume_path}' does not contain an {progress_name} checkpoint step. "
            f"Use a directory like '{progress_name}_0042', or pass --resume_start_step explicitly."
        )
    return int(match.group(1)) + 1


def start_training(args: argparse.Namespace):
    timestamp: str = resolve_training_timestamp(args, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    resume_log_timestamp: str | None = getattr(args, "resume_log_timestamp", None)
    print(f"\n🚀 Training started at {timestamp} for {args.num_episodes} episodes\n")
    logger: Logger = Logger("train_logs", timestamp)
    logger.log_configs(overwrite=resume_log_timestamp is None)
    if resume_log_timestamp is not None:
        print(f"🧾 Appending resume logs to train_logs/log_data_{timestamp}.json")

    set_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)
    progress_name: str = "update" if model_name == "mappo" else "episode"
    start_step: int = 1

    if args.resume_path is not None:
        start_step = (
            args.resume_start_step
            if args.resume_start_step is not None
            else infer_resume_start_step(args.resume_path, progress_name)
        )
        if start_step < 1:
            raise ValueError(f"--resume_start_step must be >= 1, got {start_step}")
        if start_step > args.num_episodes:
            raise ValueError(
                f"Resume start {progress_name} {start_step} exceeds target --num_episodes {args.num_episodes}."
            )
        model.load(args.resume_path)
        print(f"📥 Resuming {model_name} from {args.resume_path} at {progress_name} {start_step}")

    if model_name in ["maddpg", "matd3", "masac"]:
        train_off_policy(env, model, logger, args.num_episodes, start_episode=start_step)
    elif model_name == "mappo":
        train_on_policy(env, model, logger, args.num_episodes, start_update=start_step)
    else:  # "random"
        train_random(env, model, logger, args.num_episodes, start_episode=start_step)

    print("✅ Training Completed!\n")
    print("📊 Generating plots...")
    generate_plots_if_available(logger.json_file_path, f"train_plots/{model_name}/", "train", timestamp)


def start_testing(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Testing started at {timestamp} for {args.num_episodes} episodes\n")
    logger: Logger = Logger("test_logs", timestamp)
    logger.load_configs(args.config_path)

    set_seed(config.SEED)
    env: Env = Env()
    model: MARLModel = get_model(config.MODEL.lower())

    model.load(args.model_path)
    print(f"📥 Models loaded successfully from {args.model_path}")

    test_model(env, model, logger, args.num_episodes)

    print("✅ Testing Completed!\n")
    print("📊 Generating plots...")
    generate_plots_if_available(logger.json_file_path, f"test_plots/{config.MODEL}/", "test", timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--num_episodes", type=int, required=True)
    parent_parser.add_argument("--gpu_id", type=str, default=None, help="GPU ID to use (e.g., '0', '1')")
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.add_argument("--resume_path", type=str, default=None, help="Checkpoint directory to resume from.")
    train_parser.add_argument(
        "--resume_start_step",
        type=int,
        default=None,
        help="First episode/update to run when resume_path does not encode progress, e.g. final checkpoints.",
    )
    train_parser.add_argument(
        "--resume_log_timestamp",
        type=str,
        default=None,
        help="Existing training timestamp whose log/config/plot/checkpoint run should be reused when resuming.",
    )

    test_parser = subparsers.add_parser("test", parents=[parent_parser])
    test_parser.add_argument("--model_path", type=str, required=True)
    test_parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        start_training(args)
    elif args.mode == "test":
        start_testing(args)
    print("🎉 All done!")
