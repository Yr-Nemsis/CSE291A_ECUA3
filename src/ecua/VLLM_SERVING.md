# Serving Models with vLLM

This guide explains how to serve the UGround-V1-2B and Qwen2-VL-2B models using vLLM for use in the evaluation notebook.

## Prerequisites

- vLLM installed: `pip install vllm`
- CUDA-capable GPU(s)
- Sufficient GPU memory (each 2B model needs ~4-6GB)

## Quick Start

### Option 1: Using the provided script

```bash
# Terminal 1: Serve UGround-V1-2B
python src/ecua/serve_models_vllm.py --model uground-2b --port 8000 --trust-remote-code

# Terminal 2: Serve Qwen2-VL-2B
python src/ecua/serve_models_vllm.py --model qwen2-vl-2b --port 8001 --trust-remote-code
```

### Option 2: Using vLLM directly

```bash
# Terminal 1: Serve UGround-V1-2B
vllm serve osunlp/UGround-V1-2B --port 8000 --trust-remote-code

# Terminal 2: Serve Qwen2-VL-2B
vllm serve Qwen/Qwen2-VL-2B --port 8001 --trust-remote-code
```

## Advanced Options

### Tensor Parallelism (Multiple GPUs)

If you have multiple GPUs, you can use tensor parallelism:

```bash
python src/ecua/serve_models_vllm.py --model uground-2b --port 8000 --tensor-parallel-size 2 --trust-remote-code
```

### Custom Data Type

```bash
python src/ecua/serve_models_vllm.py --model uground-2b --port 8000 --dtype float16 --trust-remote-code
```

## Verify Servers are Running

Once started, you should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

You can test the API:
```bash
curl http://localhost:8000/v1/models
```

## Using in the Notebook

The notebook (`uground_qwen_click100k_eval.ipynb`) is configured to use:
- UGround-V1-2B at `http://localhost:8000/v1`
- Qwen2-VL-2B at `http://localhost:8001/v1`

If you use different ports, update the `API_ENDPOINTS` dictionary in the notebook.

## Troubleshooting

1. **Out of Memory**: Reduce batch size or use tensor parallelism
2. **Port Already in Use**: Change the port with `--port`
3. **Model Not Found**: Ensure you have internet access to download from HuggingFace
4. **Trust Remote Code Error**: Add `--trust-remote-code` flag

## Performance Tips

- Use `--dtype float16` or `--dtype bfloat16` for faster inference
- Use tensor parallelism if you have multiple GPUs
- Consider using quantization for lower memory usage

