#!/usr/bin/env python3
"""
EvalScope Testing Script

This script evaluates language models on various benchmarks using the evalscope
framework. Default benchmark is AIME24, but supports many datasets including
gsm8k, math_500, competition_math, and more.

Features:
- Two generation modes: Greedy (deterministic) and Sampling (stochastic)
- Thinking mode support for chain-of-thought reasoning
- Configurable batch size for parallel evaluation
- Multiple dataset support

Default model: Qwen3-0.6B at /home/models/Qwen3-0.6B
"""

import argparse
import os
import sys

# Add evalscope to path (installed as editable package in this repo)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evalscope'))

from evalscope import TaskConfig, run_task


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM on math/reasoning benchmarks using evalscope',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/models/Qwen3-0.6B',
        help='Path to the model checkpoint'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=['aime24'],
        help='Dataset(s) to evaluate on (e.g., aime24 gsm8k math_500 arc)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: outputs/{dataset}_eval)'
    )

    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        help='Enable thinking/reasoning mode (chain-of-thought)'
    )

    parser.add_argument(
        '--sampling',
        action='store_true',
        help='Use sampling instead of greedy generation'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Number of samples to evaluate (None for all)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation (number of samples processed in parallel)'
    )

    return parser.parse_args()


def get_generation_config(sampling: bool, enable_thinking: bool) -> dict:
    """
    Get generation configuration based on mode.

    Args:
        sampling: Whether to use sampling (True) or greedy (False)
        enable_thinking: Whether thinking mode is enabled

    Returns:
        Dictionary with generation parameters
    """
    if not sampling:
        # Greedy mode: deterministic
        return {
            'do_sample': False,
            'max_tokens': 4096,
        }
    elif enable_thinking:
        # Sampling with thinking: Qwen recommended params for reasoning
        return {
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'max_tokens': 30000,  # Large for extensive reasoning
        }
    else:
        # Sampling without thinking: Qwen recommended params
        return {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'max_tokens': 10000,
        }


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set output directory with dataset name if not specified
    if args.output_dir is None:
        args.output_dir = f'outputs/{"_".join(args.dataset)}_eval'

    # Get generation config based on mode
    generation_config = get_generation_config(args.sampling, args.enable_thinking)

    # Prepare model_args for thinking mode
    model_args = {}
    if args.enable_thinking:
        model_args['chat_template_kwargs'] = {'enable_thinking': True}

    # Prepare dataset_args for thinking mode (filter out thinking blocks)
    dataset_args = {}
    if args.enable_thinking:
        for ds in args.dataset:
            dataset_args[ds] = {
                'filters': {'remove_until': '</think>'}
            }

    # Build TaskConfig
    task_cfg = TaskConfig(
        model=args.model_path,
        datasets=args.dataset,
        eval_type='checkpoint',  # Direct model loading
        # eval_backend defaults to 'native' - no need to set it
        work_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        eval_batch_size=args.batch_size,
        generation_config=generation_config,
        dataset_dir='/home/data/.cache/evalscope',  # As per CLAUDE.local.md
        model_args=model_args,  # Will be empty dict if no thinking mode
        dataset_args=dataset_args,  # Will be empty dict if no thinking mode
    )

    # Print configuration summary
    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Model Path:      {args.model_path}")
    print(f"Dataset(s):      {', '.join(args.dataset)}")
    print(f"Generation Mode: {'Sampling' if args.sampling else 'Greedy'}")
    print(f"Thinking Mode:   {'Enabled' if args.enable_thinking else 'Disabled'}")
    print(f"Sample Limit:    {args.limit if args.limit else 'All'}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Random Seed:     {args.seed}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"\nGeneration Config:")
    for key, value in generation_config.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    print()

    # Run evaluation
    try:
        print("Starting evaluation...")
        results = run_task(task_cfg=task_cfg)

        print()
        print("=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {args.output_dir}")
        print(f"  - Configuration: {args.output_dir}/configs/")
        print(f"  - Reports:       {args.output_dir}/reports/")
        print(f"  - Logs:          {args.output_dir}/logs/")
        print("=" * 80)

        return results
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR DURING EVALUATION")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print(f"  - Verify model path exists: {args.model_path}")
        print(f"  - Verify dataset name is valid: {args.dataset}")
        print(f"  - Check logs in: {args.output_dir}/logs/")
        print("=" * 80)
        raise


if __name__ == '__main__':
    main()
