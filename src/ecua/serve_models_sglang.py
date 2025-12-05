#!/usr/bin/env python3
"""
Serve UGround-V1-2B and Qwen2-VL-2B models using SGLang.

Usage:
    # Serve both models on different ports
    python serve_models_sglang.py --model uground-2b --port 8000
    python serve_models_sglang.py --model qwen2-vl-2b --port 8001
"""

import argparse
import subprocess
import sys


MODEL_CONFIGS = {
    "uground-2b": "osunlp/UGround-V1-2B",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B",
}


def serve_model(model_key: str, port: int = 8000, **kwargs):
    """
    Serve a model using SGLang.
    
    Args:
        model_key: Key from MODEL_CONFIGS
        port: Port to serve on
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_id = MODEL_CONFIGS[model_key]
    
    # Build SGLang serve command
    cmd = [
        "sgl", "serve", model_id,
        "--port", str(port),
    ]
    
    # Add additional kwargs
    for key, value in kwargs.items():
        if value is not None and value is not False:
            if isinstance(value, bool):
                cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Starting SGLang server for {model_key} ({model_id}) on port {port}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Access API at: http://localhost:{port}/v1")
    print("-" * 80)
    
    # Run the server
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Serve vision-language models with SGLang")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to serve"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: bfloat16)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization ratio (0.0-1.0). Default: 0.5"
    )
    
    args = parser.parse_args()
    
    kwargs = {
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    
    serve_model(
        model_key=args.model,
        port=args.port,
        **kwargs
    )


if __name__ == "__main__":
    main()

