import os
import importlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

from utils.training_run import resolve_training_timestamp


class ResumeLoggingTests(unittest.TestCase):
    def test_resume_log_timestamp_reuses_existing_run_timestamp(self) -> None:
        args = SimpleNamespace(resume_path="saved_models/masac_old/episode_0010", resume_log_timestamp="old")

        self.assertEqual(resolve_training_timestamp(args, "new"), "old")

    def test_resume_log_timestamp_requires_resume_path(self) -> None:
        args = SimpleNamespace(resume_path=None, resume_log_timestamp="old")

        with self.assertRaises(ValueError):
            resolve_training_timestamp(args, "new")

    def test_log_configs_can_preserve_existing_file(self) -> None:
        Logger = self._load_logger_with_fake_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = Logger(tmp_dir, "old")
            with open(logger.config_file_path, "w", encoding="utf-8") as file:
                file.write("existing")

            logger.log_configs(overwrite=False)

            with open(logger.config_file_path, "r", encoding="utf-8") as file:
                self.assertEqual(file.read(), "existing")

    def test_log_configs_writes_missing_file_when_preserving(self) -> None:
        Logger = self._load_logger_with_fake_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = Logger(tmp_dir, "old")

            logger.log_configs(overwrite=False)

            self.assertTrue(os.path.exists(logger.config_file_path))

    def _load_logger_with_fake_config(self):
        fake_numpy = types.SimpleNamespace(
            ndarray=tuple,
            int32=int,
            int64=int,
            float32=float,
            float64=float,
        )
        fake_config = types.SimpleNamespace(MODEL="masac", SEED=1234)
        original_numpy = sys.modules.get("numpy")
        original_config = sys.modules.get("config")
        original_logger = sys.modules.pop("utils.logger", None)
        sys.modules["numpy"] = fake_numpy
        sys.modules["config"] = fake_config
        try:
            logger_module = importlib.import_module("utils.logger")
            return logger_module.Logger
        finally:
            if original_logger is not None:
                sys.modules["utils.logger"] = original_logger
            else:
                sys.modules.pop("utils.logger", None)
            if original_numpy is not None:
                sys.modules["numpy"] = original_numpy
            else:
                sys.modules.pop("numpy", None)
            if original_config is not None:
                sys.modules["config"] = original_config
            else:
                sys.modules.pop("config", None)


if __name__ == "__main__":
    unittest.main()
