# Serving Models with SGLang

This guide explains how to serve the UGround-V1-2B and Qwen2-VL-2B models using SGLang.

## Prerequisites

- SGLang installed: `pip install sglang[all]`
- CUDA-capable GPU(s)
- Sufficient GPU memory

## Quick Start

### Option 1: Using the provided script

```bash
# Terminal 1: Serve UGround-V1-2B
python src/ecua/serve_models_sglang.py --model uground-2b --port 8000 --trust-remote-code

# Terminal 2: Serve Qwen2-VL-2B
python src/ecua/serve_models_sglang.py --model qwen2-vl-2b --port 8001 --trust-remote-code
```

### Option 2: Using SGLang directly

```bash
# Terminal 1: Serve UGround-V1-2B
sgl serve osunlp/UGround-V1-2B --port 8000 --trust-remote-code

# Terminal 2: Serve Qwen2-VL-2B
sgl serve Qwen/Qwen2-VL-2B --port 8001 --trust-remote-code
```

## Basic Command Structure

```bash
sgl serve <model_path_or_id> [options]
```

## Common Options

- `--port <port>`: Port to serve on (default: 30000)
- `--host <host>`: Host to bind to (default: 0.0.0.0)
- `--tensor-parallel-size <n>`: Number of GPUs for tensor parallelism
- `--dtype <dtype>`: Data type (float16, bfloat16, float32)
- `--trust-remote-code`: Trust remote code from HuggingFace
- `--gpu-memory-utilization <ratio>`: GPU memory utilization (0.0-1.0)

## Example with Memory Control

If you're running into GPU memory issues:

```bash
sgl serve osunlp/UGround-V1-2B \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 \
    --dtype float16
```

## Verify Server is Running

Once started, you should see output indicating the server is ready. You can test the API:

```bash
curl http://localhost:8000/v1/models
```

## Using in the Notebook

The notebook (`uground_qwen_click100k_eval.ipynb`) can be configured to use SGLang servers by updating the `API_ENDPOINTS`:

```python
API_ENDPOINTS = {
    "uground-2b": "http://localhost:8000/v1",
    "qwen2-vl-2b": "http://localhost:8001/v1",
}
```

SGLang provides OpenAI-compatible API endpoints, so the notebook code should work without changes.

## Troubleshooting

1. **SGLang not found**: Install with `pip install sglang[all]`
2. **Out of Memory**: Reduce `--gpu-memory-utilization` or use `--dtype float16`
3. **Port Already in Use**: Change the port with `--port`
4. **Model Not Found**: Ensure you have internet access to download from HuggingFace

## Performance Tips

- Use `--dtype float16` for faster inference and lower memory
- Use tensor parallelism if you have multiple GPUs
- SGLang is optimized for high-throughput scenarios

## Note on Vision Models

SGLang may have varying support for vision-language models. If you encounter issues, you may need to:
- Check SGLang documentation for vision model support
- Use vLLM as an alternative (see `VLLM_SERVING.md`)
- Use the direct transformers approach in the notebook

