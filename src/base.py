import logging
import os
import sys, subprocess
from typing import Any, List, Optional
from src.ensemble.ensemble_factory import EnsembleFactory
from src.evaluation.evaluation_base import EvaluationBase
from src.evaluation.evaluation_factory import EvaluationFactory
from src.models.llm_judge import LLMJudge
from src.models.vllm import VLLM
from src.preprocessing.data_module import DataModule
from src.strategies.strategy_factory import StrategyFactory
from src.technical.utils import (
    get_results_directory,
    get_dataset_config,
    set_all_seeds,
    get_eval_config_from_path,
)
from src.technical.configs.evaluation_config import EvaluationConfig
import pathlib


class FullPipeline:
    """
    The primary entry point for the library. Integrates data acquisition,
    model inference, ensemble logic, and evaluation into a unified interface.
    """

    def __init__(self):
        """
        Initializes the FullPipeline with a logger and a placeholder for
        background processes (e.g., the Streamlit visualizer).
        """
        self.logger = logging.getLogger(__name__)
        self._proc = None

    def prepare_data(self, download: bool = False) -> None:
        """
        Handles data acquisition and the complete preprocessing pipeline.

        Ensures data is structured for downstream modules by creating a
        DataModule instance and executing the preprocessing flow.

        Args:
            download (bool): If True, downloads the dataset from Hugging Face.
                Requires HF_API_TOKEN if accessing gated datasets.
        """
        data_module = DataModule(load_from_hf=download)

        data_module.run()

    def run_experiment(
        self,
        dataset_name: str,
        strategy_name: str,
        model_name: str,
        model_object: Optional[VLLM] = None,
        restart_problem_id: Optional[str] = None,
        restart_version: Optional[str] = None,
        param_set_number: Optional[int] = None,
        prompt_number: Optional[int] = 1,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Executes a single experiment strategy for a specific model and dataset.

        This method handles the lifecycle of the experiment, including model
        initialization via vLLM, directory setup, and execution of strategies
        (e.g., Direct, Descriptive, Contrastive, or Classification).

        Args:
            dataset_name (str): Name of the dataset to use.
            strategy_name (str): Name of the strategy to execute.
            model_name (str): Name of the model (must be compatible with vLLM).
            model_object (Optional[VLLM]): An existing VLLM instance. If provided,
                skips new model initialization.
            restart_problem_id (Optional[str]): Specific problem ID to resume from.
            restart_version (Optional[str]): Version to restart; defaults to "latest".
            param_set_number (Optional[int]): Index of parameters to pull from config.
            prompt_number (Optional[int]): Version of the prompt template to use.
            seed (Optional[int]): Random seed for reproducibility.

        Raises:
            RuntimeError: If the model fails to initialize.
        """
        if seed:
            set_all_seeds(seed)
        self.logger.info(
            f"Creating strategy '{strategy_name}' for dataset '{dataset_name}' with model '{model_name}'"
        )

        try:
            target_version = (
                restart_version
                if (restart_version and restart_version.strip())
                else "latest"
            )

            results_dir = get_results_directory(
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                model_name=model_name,
                version=target_version,
                create_dir=True,
            )

            strategy_factory = StrategyFactory()

            model = model_object
            if not model:
                model = self._load_model(
                    model_name=model_name, param_set_number=param_set_number
                )

            if model is None:
                raise RuntimeError(f"Failed to initialize model: {model_name}")

            strategy = strategy_factory.create_strategy(
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                model_object=model,
                results_dir=results_dir,
                param_set_number=param_set_number,
                prompt_number=prompt_number,
            )

            self.logger.info("Strategy created successfully. Running experiment...")
            strategy.run(restart_problem_id=restart_problem_id)
            self.logger.info(
                f"Experiment run complete for {dataset_name} / {strategy_name}."
            )

            if model_object is None:
                model.stop()

        except Exception as e:
            self.logger.error(
                f"An error occurred during the experiment run: {e}", exc_info=True
            )
            if model is not None and hasattr(model, "stop"):
                model.stop()
            raise e

    def run_ensemble(
        self,
        dataset_name: str,
        members_configuration: List[List[str]],
        type_name: str,
        vllm_model_name: Optional[str] = None,
        vllm_param_set_number: Optional[int] = None,
        llm_model_name: Optional[str] = None,
        llm_param_set_number: Optional[int] = None,
        model_object: Optional[VLLM] = None,
        prompt_number: Optional[int] = 1,
        version: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Aggregates results from multiple models/strategies into an ensemble.

        Supports various aggregation methods including majority voting, confidence
        scores, and reasoning-based judging (using an LLM or VLM as a judge).

        Args:
            dataset_name (str): Name of the dataset.
            members_configuration (List[List[str]]): List of members, where each
                inner list is [strategy_name, model_name, version].
            type_name (str): Type of ensemble logic (e.g., 'reasoning', 'reasoning_with_image').
            vllm_model_name (Optional[str]): VLM judge name for vision-based ensembles.
            llm_model_name (Optional[str]): LLM judge name for reasoning-based ensembles.
            model_object (Optional[VLLM]): An existing model instance to use as a judge.
            prompt_number (Optional[int]): Version of the ensemble prompts to use.
            version (Optional[int]): Specific version of results to ensemble.
            seed (Optional[int]): Random seed for reproducibility.
        """
        if seed:
            set_all_seeds(seed)
        self.logger.info(
            f"Creating ensemble '{type_name}' for dataset '{dataset_name}' with members: {members_configuration}')"
        )
        try:
            ensemble_factory = EnsembleFactory()

            if not model_object:
                if type_name == "reasoning_with_image" and vllm_model_name:
                    self.logger.info(
                        f"Initializing VLLM model '{vllm_model_name}' for reasoning with image ensemble."
                    )
                    model = VLLM(
                        model_name=vllm_model_name,
                        param_set_number=vllm_param_set_number,
                    )

                elif (
                    get_dataset_config(dataset_name).category == "BP" and llm_model_name
                ) or (type_name == "reasoning" and llm_model_name):
                    self.logger.info(
                        f"Initializing LLM model '{llm_model_name}' for ensemble."
                    )

                    model = LLMJudge(
                        model_name=llm_model_name, param_set_number=llm_param_set_number
                    )
                else:
                    model = None
            else:
                model = model_object

            ensemble = ensemble_factory.create_ensemble(
                dataset_name=dataset_name,
                members_configuration=members_configuration,
                skip_missing=True,
                judge_model=model,
                type_name=type_name,
                prompt_number=prompt_number,
                version=version,
                seed=seed,
            )

            self.logger.info("Ensemble created successfully. Running ensemble...")
            ensemble.evaluate()
            self.logger.info(f"Ensemble run complete for {dataset_name} / {type_name}.")

            if model and model_object is None:
                model.stop()

        except ImportError as e:
            self.logger.error(
                f"Failed to create ensemble. Does '{type_name}' exist and is it importable? Error: {e}",
                exc_info=True,
            )
            if model_object is None:
                model.stop()
            sys.exit(1)
        except Exception as e:
            self.logger.error(
                f"An error occurred during the experiment run: {e}", exc_info=True
            )
            if model_object is None:
                model.stop()
            raise e

    def run_missing_evaluations_in_directory(
        self,
        path: str,
        judge_model_name: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.3",
        judge_param_set_number: Optional[int] = 1,
        prompt_number: int = 1,
        seed: Optional[int] = 42,
    ):
        """
        Recursively scans a results directory and executes evaluations for
        any experiment folders missing results.

        This is a utility method to ensure all completed experiments have
        corresponding evaluation metrics without manual triggering.

        Args:
            path (str): Root directory to scan (must start with 'results/').
            judge_model_name (str): The model to use as the evaluation judge.
            judge_param_set_number (int): Parameter set for the judge model.
            prompt_number (int): Version of the evaluation prompt.
            seed (int): Random seed for the evaluation judge.
        """
        root_path = pathlib.Path(path)
        if not str(root_path).startswith("results"):
            self.logger.info(f"Skipping: Path '{path}' must start with 'results/'.")
            return

        if not root_path.exists() or not root_path.is_dir():
            self.logger.info(
                f"Skipping: Path '{path}' does not exist or is not a directory."
            )
            return

        self.logger.info(f"Scanning for missing evaluations in: {root_path}")

        # .rglob('*') finds all files and directories recursively
        for subdir in root_path.rglob("*"):

            if subdir.is_dir() and ("ver" in subdir.name):

                subfolders = [f for f in subdir.iterdir() if f.is_dir()]
                if len(subfolders) > 0:
                    continue

                eval_file_exists = next(subdir.glob("evaluation_results*"), None)

                if eval_file_exists:
                    self.logger.info(
                        f"Found existing results ({eval_file_exists.name}) in {subdir}. Skipping."
                    )
                    continue

                is_ensemble = "ensemble" in str(subdir).lower()

                try:
                    config = get_eval_config_from_path(
                        path=str(subdir),
                        ensemble=is_ensemble,
                        judge_model_name=judge_model_name,
                        prompt_number=prompt_number,
                        param_set_number=judge_param_set_number,
                    )

                    self.logger.info(f"Running evaluation for: {subdir}")
                    self.run_evaluation(config=config, seed=seed)

                except ValueError as e:
                    self.logger.info(f"Could not parse config for {subdir}: {e}")

    def run_evaluation(
        self,
        config: EvaluationConfig,
        evaluator: Optional[EvaluationBase] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Evaluates the performance of a single model or ensemble experiment.

        Calculates basic metrics and uses either a key-based comparison or
        a judge model (for open-ended tasks) to verify answers.

        Args:
            config (EvaluationConfig): Configuration object describing the target results.
            evaluator (Optional[EvaluationBase]): An existing evaluator instance
                to reduce overhead for sequential runs.
            seed (Optional[int]): Random seed for the evaluator.
        """
        if seed:
            set_all_seeds(seed)

        if evaluator is not None:
            self.logger.info("Using provided evaluator instance.")
            stop_after_evaluation = False
        else:
            self.logger.info("Creating evaluator instance from configuration.")
            stop_after_evaluation = True
            eval_factory = EvaluationFactory()

            evaluator = eval_factory.create_evaluator(
                dataset_name=config.dataset_name,
                ensemble=config.ensemble,
                strategy_name=config.strategy_name,
                judge_model_object=config.judge_model_object,
                judge_model_name=config.judge_model_name,
                judge_param_set_number=config.judge_param_set_number,
                prompt_number=config.prompt_number,
            )

        evaluator.run_evaluation(
            dataset_name=config.dataset_name,
            version=config.version,
            strategy_name=config.strategy_name,
            model_name=config.model_name,
            ensemble=config.ensemble,
            type_name=config.type_name,
            evaluation_output_path=config.evaluation_output_path,
            concat=config.concat,
            output_all_results_concat_path=config.output_all_results_concat_path,
        )

        if stop_after_evaluation and evaluator.judge_model_object is not None:
            evaluator.judge_model_object.stop()

    def visualise(
        self, csv_path: str = os.path.join("results", "all_results_concat.csv")
    ) -> None:
        """
        Launches the interactive Streamlit dashboard.

        Provides a user-friendly UI for exploring metrics, comparing model
        performance, and browsing experiment results. Starts as a background process.

        Args:
            csv_path (str): Path to the concatenated results CSV file.
        """
        visualiser_path = os.path.join("src", "visualisation", "visualiser.py")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", visualiser_path, "--", csv_path]
        )
        self.logger.info(f"Streamlit visualiser started with PID: {self._proc.pid}")

    def stop_visualiser(self):
        """
        Terminates the Streamlit background process.

        Closing the browser tab does not stop the Python process; this method
        ensures the process is cleanly killed and resources are freed.
        """
        if self._proc and self._proc.poll() is None:
            self.logger.info(
                f"Terminating Streamlit visualiser with PID: {self._proc.pid}"
            )
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "Streamlit visualiser did not terminate in time. Killing process."
                )
                self._proc.kill()
            self.logger.info("Streamlit visualiser terminated.")

    def _load_model(
        self, model_name: str, param_set_number: Optional[int] = None
    ) -> Any:

        self.logger.info(f"Attempting to load model: '{model_name}'")

        try:
            vllm_model = VLLM(model_name=model_name, param_set_number=param_set_number)

            if vllm_model:
                return vllm_model
            else:
                return None

        except TimeoutError as e:
            self.logger.critical(
                f"Failed to start VLLM server for model '{model_name}'. "
                f"Pipeline execution cannot continue. Error: {e}"
            )
            return None

        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during VLLM setup for '{model_name}'. Error: {e}"
            )
            return None

    def check_data_preprocessed(self, dataset_name: str) -> bool:
        """
        Validates if a dataset is correctly preprocessed and formatted.

        Checks for the existence of the required directory structure:
        - data/<dataset_name>/problems/
        - data/<dataset_name>/jsons/

        Args:
            dataset_name (str): The name of the dataset to verify.

        Returns:
            bool: True if the structure is valid and contains data, False otherwise.
        """
        self.logger.info(
            f"Checking for preprocessed data for dataset: {dataset_name}..."
        )
        base_data_path = os.path.join("data", dataset_name)
        problems_path = os.path.join(base_data_path, "problems")
        jsons_path = os.path.join(base_data_path, "jsons")

        if not os.path.exists(base_data_path):
            self.logger.error(f"Data directory not found: {base_data_path}")
            return False

        if not os.path.exists(problems_path):
            self.logger.error(
                f"Standardized 'problems' directory not found: {problems_path}"
            )
            return False

        if not os.path.exists(jsons_path):
            self.logger.error(f"Standardized 'jsons' directory not found: {jsons_path}")
            return False

        if not any(fname.endswith(".json") for fname in os.listdir(jsons_path)):
            self.logger.error(f"No JSON metadata files found in: {jsons_path}")
            return False

        self.logger.info(f"Found preprocessed data at: {base_data_path}")
        return True
