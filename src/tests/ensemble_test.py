import argparse
import json
import sys
import logging

from src.models.vllm import VLLM
from src.technical.utils import get_dataset_config
from src.base import FullPipeline
from src.models.vllm import VLLM


logger = logging.getLogger(__name__)


def json_list(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("members_configuration must be valid JSON")


def seed_and_version_test(dataset):
    pipeline = FullPipeline()
    ### top overall:
    top_members_overall = [
        ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
        ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
        ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
        ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
        ["direct", "OpenGVLab/InternVL3-8B", "1"],
    ]
    configurations = [top_members_overall]

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    type_names = ["majority"]
    versions = [100, 101, 102]
    seeds = [100, 100, 101]

    dataset_config = get_dataset_config(dataset)
    for ver, seed in zip(versions, seeds):
        for config in configurations:
            if dataset_config.category == "BP":
                model = VLLM(model_name)
            else:
                model = None
            for type_name in type_names:
                pipeline.run_ensemble(
                    dataset_name=dataset,
                    members_configuration=config,
                    type_name=type_name,
                    model_object=model,
                    version=ver,
                    seed=seed,
                )
            if model:
                model.stop()


def get_config_and_seed(version: int, dataset_name: str, model_number: int = 0):
    """
    Returns a specific configuration and seed based on the version number.

    Args:
        version (int): Number of version
        dataset_name (str): Name of the dataset (bp, cvr, raven, marsvqa)
        model_number (int): Number of the model to use (default is 1)
    Returns:
        tuple: (list of ensemble members, int seed)
    """

    seed = version % 10

    top_members_dataset = []
    if "bp" in dataset_name.lower():
        top_members_dataset = [
            ["direct", "OpenGVLab/InternVL3-8B", "1"],
            ["direct", "OpenGVLab/InternVL3-8B", "3"],
            ["descriptive", "OpenGVLab/InternVL3-8B", "1"],
            ["descriptive", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
        ]
    elif dataset_name == "cvr":
        top_members_dataset = [
            ["classification", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["direct", "OpenGVLab/InternVL3-8B", "1"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["classification", "OpenGVLab/InternVL3-8B", "3"],
        ]
    elif dataset_name == "raven":
        top_members_dataset = [
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["contrastive", "OpenGVLab/InternVL3-8B", "3"],
        ]
    elif dataset_name == "marsvqa":
        top_members_dataset = [
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["descriptive", "OpenGVLab/InternVL3-8B", "1"],
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    mapping = {
        0: [
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["direct", "OpenGVLab/InternVL3-8B", "1"],
        ],  # top_members_overall
        1: top_members_dataset,
        2: [
            ["classification", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["descriptive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
        ],  # qwen
        3: [
            ["classification", "llava-hf/llava-v1.6-mistral-7b-hf", "3"],
            ["contrastive", "llava-hf/llava-v1.6-mistral-7b-hf", "3"],
            ["descriptive", "llava-hf/llava-v1.6-mistral-7b-hf", "3"],
            ["direct", "llava-hf/llava-v1.6-mistral-7b-hf", "1"],
        ],  # llava
        4: [
            ["classification", "OpenGVLab/InternVL3-8B", "1"],
            ["contrastive", "OpenGVLab/InternVL3-8B", "3"],
            ["descriptive", "OpenGVLab/InternVL3-8B", "1"],
            ["direct", "OpenGVLab/InternVL3-8B", "1"],
        ],  # intern
        5: [
            ["direct", "OpenGVLab/InternVL3-8B", "1"],
            ["direct", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["direct", "llava-hf/llava-v1.6-mistral-7b-hf", "1"],
        ],  # direct
        6: [
            ["descriptive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["descriptive", "llava-hf/llava-v1.6-mistral-7b-hf", "3"],
            ["descriptive", "OpenGVLab/InternVL3-8B", "1"],
        ],  # descriptive
        7: [
            ["contrastive", "OpenGVLab/InternVL3-8B", "3"],
            ["contrastive", "Qwen/Qwen2.5-VL-7B-Instruct", "3"],
            ["contrastive", "llava-hf/llava-v1.6-mistral-7b-hf", "3"],
        ],  # contrastive
        8: [
            ["classification", "OpenGVLab/InternVL3-8B", "1"],
            ["classification", "Qwen/Qwen2.5-VL-7B-Instruct", "1"],
            ["classification", "llava-hf/llava-v1.6-mistral-7b-hf", "3"],
        ],  # classification
    }

    group_index = version // 10 - model_number * 10

    return mapping[group_index], seed


def run_pipeline_for_dataset(dataset: str, pipeline, model_name, model_number=0):
    """
    Runs the ensemble pipeline and resets the model every 2 runs that require model.

    Args:
        dataset (str): Name of the dataset.
        pipeline: The pipeline object.
        model_factory: A function or class that returns a fresh model instance.
    """
    type_names = ["reasoning", "reasoning_with_image", "majority", "confidence"]

    # Initialize the first model instance
    model = VLLM(model_name)
    run_counter = 0

    offset = model_number * 100

    dataset_config = get_dataset_config(dataset)

    for type_name in type_names:
        if (
            type_name in ["reasoning", "reasoning_with_image"]
            or dataset_config.category == "BP"
        ):
            max_range = 80 if dataset_config.category == "BP" else 90
            versions_to_run = (v + offset for v in range(0, max_range, 10))
        else:
            versions_to_run = (v + offset for v in range(90))

        for ver in versions_to_run:
            config, seed = get_config_and_seed(ver, dataset, model_number)

            pipeline.run_ensemble(
                dataset_name=dataset,
                members_configuration=config,
                type_name=type_name,
                model_object=model,
                version=ver,
                seed=seed,
            )

            # Increment counter and check for reset
            if (
                type_name in ["reasoning", "reasoning_with_image"]
                or dataset_config.category == "BP"
            ):
                run_counter += 1
                if run_counter % 2 == 0:
                    print(f"--- Resetting model after {run_counter} total runs ---")
                    model.stop()
                    model = VLLM(model_name)  # Replace old model with a fresh one


def run_pipeline_for_dataset_test(dataset: str, pipeline, model_name, model_number=0):
    type_names = ["reasoning", "reasoning_with_image", "majority", "confidence"]

    model = VLLM(model_name)
    run_counter = 0
    offset = model_number * 100
    dataset_config = get_dataset_config(dataset)

    for type_name in type_names:
        is_singular_run = (
            type_name in ["reasoning", "reasoning_with_image"]
            or dataset_config.category == "BP"
        )

        versions_to_run = []

        if is_singular_run:
            versions_to_run = [0, 10]

            if "raven" in dataset.lower() and type_name == "reasoning_with_image":
                if model_number == 1:
                    versions_to_run.append(30)
        else:
            versions_to_run = list(range(0, 20))

        final_versions = [v + offset for v in versions_to_run]

        for ver in final_versions:
            config, seed = get_config_and_seed(ver, dataset, model_number)

            pipeline.run_ensemble(
                dataset_name=dataset,
                members_configuration=config,
                type_name=type_name,
                model_object=model,
                version=ver,
                seed=seed,
            )

            # Reset logic for models (only applies to reasoning/BP runs)
            if is_singular_run:
                run_counter += 1
                if run_counter % 2 == 0:
                    print(f"--- Resetting model after {run_counter} total runs ---")
                    model.stop()
                    model = VLLM(model_name)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single ensemble experiment")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use (same as in dataset_config.json)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    dataset = args.dataset_name
    pipeline = FullPipeline()
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_number = 0
    # model_name = "OpenGVLab/InternVL3-8B"
    # model_number = 1

    # run_pipeline_for_dataset(dataset, pipeline, model_name, model_number)
    run_pipeline_for_dataset(dataset, pipeline, model_name, model_number)

    # run_all_ensembles(dataset)
    # seed_and_version_test(dataset) PASSED - models always deterministic, no model changes with seed
