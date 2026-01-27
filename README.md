# IQ-Bench: VLLM Evaluation Framework

IQ-Bench is a modular Python package designed to streamline the evaluation of Vision Language Models (VLMs) on visual analogy tasks.


## Key Functionalities

* **Unified Pipeline**: The `FullPipeline` class serves as the single entry point to consolidate data processing, model inference, and evaluation.
* **Diverse Reasoning Strategies**: Supports multiple execution methods.
* **Ensemble Capabilities**: Allows for model committees that aggregate results.
* **Interactive Visualization**: Includes a built-in Streamlit dashboard for browsing metrics and model performance.
* **Automated Evaluation**: Supports both close-ended and open-ended results evaluations. Features a "judge" model to validate results on open-ended tasks where simple key comparison is insufficient.

## Installation & Requirements

### System Requirements
* **OS**: Linux-based environments are required. Windows users must utilize WSL2.
* **Hardware**: Users must estimate required resources (GPU, VRAM etc.) based on the specific VLM models they intend to deploy.

Install the package directly via PyPI:
```bash
pip install iqbench
```

## Available Datasets
The framework currently supports 4 datasets for running experiments. These datasets capture a broad visual spectrum covering different types of reasoning tasks, including both open-ended and close-ended tasks.

### VCog-Bench
VCog-Bench is a publicly available, zero-shot abstract visual reasoning benchmark designed to evaluate Multimodal Large Language Models. It integrates established AVR datasets and is available on the Hugging Face platform. This framework supports experiments on all subsets: **Raven Progressive Matrices** *(dataset_name = "raven")*, **CVR** *(dataset_name = "cvr")*, **MaRs-VQA** *(dataset_name = "marsvqa")*.    
**Source**: [VCog-Bench Dataset](https://huggingface.co/datasets/vcog/vcog-bench)

 
### Bongard Problems
The Bongard Problems dataset is a classic collection of visual reasoning puzzles introduced by Mikhail Bongard. Each problem consists of two sets of images (typically 6 on the left and 6 on the right), where all images in one set share an abstract visual rule that the other set does not. The task is to identify the rule that distinguishes the two sets-such as differences in shape, topology, symmetry, count, or spatial relations-without being explicitly told what features matter.  
**Availability**: *Not available on Hugging Face.* Users must provide this dataset manually in the `data_raw` directory. Data can be found in the following sources: 
  * [BP Image Repository](https://github.com/XinyuYun/bongard-problems)
  * [Bongard in Wonderland (Solutions)](https://github.com/ml-research/bongard-in-wonderland)


## Supported Models
The framework supports all Vision Language Models (VLMs) compatible with the **vLLM** package ([package documentation](https://docs.vllm.ai/en/latest/)). To use a specific model, user must define its attributes and parameters in a JSON configuration file. The path to this file must be provided as an environment variable `MODELS_CONFIG_JSON_PATH`. 
The following models are pre-configured within the package, allowing for immediate deployment without additional manual configuration:
* **InternVL Series (OpenGVLab)**:
    * InternVL3-8B
    * InternVL3-14B
    * InternVL3-38B
    * InternVL3-78B
* **Qwen Series**:
    * Qwen2.5-VL-3B-Instruct
    * Qwen2.5-VL-7B-Instruct
    * Qwen2.5-VL-32B-Instruct
    * Qwen2.5-VL-72B-Instruct
* **LLaVA Series**:
    * llava-v1.6-mistral-7b-hf
    * llava-onevision-qwen2-72b-ov-hf
* **Judge LLMs (Evaluation Judges)**:
    * Mistral-7B-Instruct-v0.3
    * Phi-3.5-mini-instruct

### Sample JSON Configuration Setup
Below is a sample configuration file structure.

```json
{
  "OpenGVLab/InternVL3-8B": {
    "model_class": "VLLM",
    "max_tokens_limit": 32000,
    "num_params_billions": 8,
    "gpu_split": false,
    "param_sets": {
      "1": {
        "temperature": 0.5,
        "max_tokens": 16384,
        "max_output_tokens": 2048,
        "limit_mm_per_prompt": 2,
        "cpu_local_testing": false,
        "custom_args": {
          "tensor_parallel_size": 1,
          "gpu_memory_utilization": 0.9
        }
      },
      "2": {
        "temperature": 0.5, ...
        }
      }
    }
  }
}
```

## Available Strategies  
*Disclaimer: Descriptions of the strategies are of an illustrative nature. For detailed descriptions on how each one works (for a specific dataset) please refer to our paper.*

### 1. Direct Strategy
* **Method:** The model is presented with the entire problem at once.
* **Process:** Model is directly asked to solve the provided puzzle.
* **Goal:** Solve the puzzle in one step based on all available visual information.
* **Argument value**: `direct`

### 2. Descriptive Strategy
* **Method:** Relies on image-to-text translation.
* **Process:** The model describes each choice image individually. These descriptions are concatenated and combined with the task description.
* **Goal:** Solve the puzzle based solely on the generated text descriptions rather than the original images.
* **Argument value**: `descriptive`

### 3. Contrastive Strategy
* **Method:** Focuses on relational differences.
* **Process:** The model is prompted to describe differences (across rows/columns or between pairs of images) iteratively.
* **Goal:** Use the identified differences and the task description to deduce the correct answer.
* **Argument value**: `contarstive`

### 4. Classification Strategy
* **Method:** Reframes the puzzle as a selection task.
* **Process:** Multiple versions of the completed problem are generated (one for each possible answer).
* **Goal:** The model evaluates each version and selects the one that best preserves the logic or assumptions of the task.
* **Argument value**: `classification`


## Available Ensembling Strategies
*Disclaimer: Descriptions of the ensembling strategies are of an illustrative nature. For detailed descriptions on how each one works please refer to our paper.*

### 1. Majority Ensemble
* **Mechanism:** Uses a voting-based system.
* **Process:** For closed-ended problems, it selects the answer that appears most frequently. For Bongard Problems (BP), an LLM synthesizes a consensus from the various proposed answers.
* **Argument value**: `majority`

### 2. Confidence Ensemble
* **Mechanism:** Prioritizes the most "certain" predictions.
* **Process:** It selects the answer with the highest average model confidence score. In specific categories, an LLM evaluates the validity of answers by weighing them against their associated confidence metrics.
* **Argument value**: `confidence`

### 3. Reasoning Ensemble
* **Mechanism:** Leverages an LLMJ udge for qualitative analysis.
* **Process:** The aggregator model analyzes not just the final answers, but the underlying reasoning chains provided by all ensemble members to determine the most logical solution.
* **Argument value**: `reasoning`

### 4. Reasoning Ensemble with Image
* **Mechanism:** Multi-modal reasoning aggregation.
* **Process:** This extends the standard Reasoning Ensemble by providing the aggregator VLLM with the original question image alongside the text-based reasoning of the candidates to improve context and visual awareness.
* **Argument value**: `reasoning_image`

## Basic Usage

### 1. Initialize the Pipeline
The `FullPipeline` class serves as the primary entry point for the library, integrating all available modules into a unified interface.

```python
from iqbench import FullPipeline
pipeline = FullPipeline()
```

### 2. Prepare Data
Handles data acquisition and the complete preprocessing pipeline. Ensures data is structured for downstream modules by creating a DataModule instance and executing the preprocessing flow.
Arguments:
  * `download (bool)`: If True, downloads the dataset from Hugging Face. Requires HF_API_TOKEN if accessing gated datasets.

```python
pipeline.prepare_data(download=True)
```

### 3. Run a single model experiment
Executes a single experiment strategy for a specific model and dataset. This method handles the lifecycle of the experiment, including model initialization via vLLM, directory setup, and execution of strategies (direct, descriptive, contrastive, or classification).
Arguments:
  * `dataset_name (str)`: Name of the dataset to use.
  * `strategy_name (str)`: Name of the strategy to execute.
  * `model_name (str)`: Name of the model (must be compatible with vLLM).
  * `model_object (Optional[VLLM])`: An existing VLLM instance. If provided, skips new model initialization.
  * `restart_problem_id (Optional[str])`: Specific problem ID to resume from.
  * `restart_version (Optional[str])`: Version to restart; defaults to "latest".
  * `param_set_number (Optional[int])`: Index of parameters to pull from config.
  * `prompt_number (Optional[int])`: Version of the prompt template to use.
  * `seed (Optional[int])`: Random seed for reproducibility.

```python
pipeline.run_experiment(
    dataset_name="cvr",
    strategy_name="direct",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    param_set_number=1,
    prompt_number=1
)
```

### 4. Run ensemble experiment
Aggregates results from multiple models/strategies into an ensemble. Supports various aggregation methods including majority voting, confidence scores, and reasoning-based judging (using an LLM or VLM as a judge).
Arguments:
* `dataset_name (str)`: Name of the dataset.
* `members_configuration (List[List[str]])`: List of members, where each inner list is `[strategy_name, model_name, version]`.`
* `type_name (str)`: Type of ensemble logic (e.g., 'reasoning', 'reasoning_with_image').
* `vllm_model_name (Optional[str])`: VLM judge name for vision-based ensembles.
* `llm_model_name (Optional[str])`: LLM judge name for reasoning-based ensembles.
* `model_object (Optional[VLLM])`: An existing model instance to use as a judge.
* `prompt_number (Optional[int])`: Version of the ensemble prompts to use.
* `version (Optional[int])`: Specific version of results to ensemble.
* `seed (Optional[int])`: Random seed for reproducibility.

```python
pipeline.run_ensemble(
    dataset_name="cvr",
    members_configuration=[["direct", "Qwen/Qwen2.5-VL-3B-Instruct", "1"], ["classification", "OpenGVLab/InternVL3-8B", "1"]],
    type_name="reasoning",
    vllm_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    llm_model_name="Qwen/Qwen2.5-VL-3B-Instruct"
)
```

### 5. Evaluate

#### 5.1 Run evaluation
Evaluates the performance of a single model or ensemble experiment. Calculates basic metrics and uses either a key-based comparison or a judge model (for open-ended tasks) to verify answers.
Arguments:
* `config (EvaluationConfig)`: Configuration object describing the target results.
* `evaluator (Optional[EvaluationBase])`: An existing evaluator instance to reduce overhead for sequential runs.
* `seed (Optional[int])`: Random seed for the evaluator.

```python
from iqbench.technical.configs import EvaluationConfig

# for running single experiment evaluation
eval_config = EvaluationConfig(
    dataset_name="cvr",
    version="1",
    strategy_name="direct",
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    ensemble=False,
    type_name=None,
    evaluation_output_path="evaluation_results",
    concat=True,
    output_all_results_concat_path="all_results_concat",
    judge_model_name="mistralai/Mistral-7B-Instruct-v0.3",
    judge_param_set_number=None,
    prompt_number=1
)

pipeline.run_evaluation(eval_config)
```

#### 5.2 Run missing evaluations 
Recursively scans a results directory and executes evaluations for any experiment folders missing results. This is a utility method to ensure all completed experiments have corresponding evaluation metrics without manual triggering.
Arguments:
  * `path (str)`: Root directory to scan (must start with 'results/').
  * `judge_model_name (str)`: The model to use as the evaluation judge.
  * `judge_param_set_number (int)`: Parameter set for the judge model.
  * `prompt_number (int)`: Version of the evaluation prompt.
  * `seed (int)`: Random seed for the evaluation judge.

```python
# for running evaluation for all experiments present in the provided directory path
pipeline.run_missing_evaluations_in_directory(path=”results”)
```

### 6. Visualise
Launches the interactive Streamlit dashboard. Provides a user-friendly UI for exploring metrics, comparing model performance, and browsing experiment results. Starts as a background process.
Arguments:
* `csv_path (str)`: Path to the concatenated results CSV file.

```python
pipeline.visualise()

# visualisation process can be stopped by using the following method
pipeline.stop_visualiser()
```