import re
import time
import uuid
import asyncio
from typing import List, Optional
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq


# ============================
# 1. Load multiple VLM models
# ============================
#
# We support hosting multiple models behind the same
# OpenAI-compatible /v1/chat/completions endpoint.
# The concrete model is selected via ChatRequest.model.
#
# You can change the IDs below to the exact checkpoints
# you want to use.

MODEL_CONFIGS = {
    # UGround 2B
    "uground-2b": "osunlp/UGround-V1-2B",
    # Qwen2-VL 2B
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B",
}

DEFAULT_MODEL_KEY = "uground-2b"


def _load_all_models():
    """
    Load all processors + models once on CUDA.
    Returns:
        processors: dict[str, AutoProcessor]
        models: dict[str, AutoModelForVision2Seq]
    """
    processors = {}
    models = {}
    for key, model_id in MODEL_CONFIGS.items():
        proc = AutoProcessor.from_pretrained(model_id)
        mdl = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        mdl.eval()
        processors[key] = proc
        models[key] = mdl
    return processors, models


PROCESSORS, MODELS = _load_all_models()

# Thread pool for running blocking model inference
_inference_executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent GPU inference


# =====================
# 2. Core UI-TARS call
# =====================
def build_uground_messages(description: str):
    """
    Build chat messages in the format expected by UI-TARS / Qwen2-VL-style processors.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # actual pixels passed via images=[...]
                {
                    "type": "text",
                    "text": f"""
Your task is to identify the precise coordinates (x, y) of a specific area/element/object on the screen based on the description.

- Your answer MUST be a single string in the format (x, y).
- x and y MUST be integers in the range [0, 1000).
- The coordinates are defined in a 1000x1000 coordinate system, where (0, 0) is the top-left and (999, 999) is the bottom-right.
- Aim to click at the center or a representative point of the described element.

Description: {description}

Answer:"""
                },
            ],
        },
    ]


def call_ui_tars_raw(
    image: Image.Image,
    query: str,
    processor: AutoProcessor,
    model: AutoModelForVision2Seq,
) -> tuple[str, int, int]:
    """
    Send one screenshot + grounding query to UI-TARS, return:
    (raw_text_response, orig_width, orig_height).
    """
    # Image is already loaded as PIL.Image
    img = image
    orig_w, orig_h = img.size

    # Build messages for processor
    messages = build_uground_messages(query)

    # Build chat prompt string (text only)
    chat_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Turn prompt + processed image into model inputs
    inputs = processor(
        text=[chat_prompt],
        images=[img],
        return_tensors="pt",
    )

    # Move to CUDA (model is on CUDA)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        # Ensure inputs don't have cache-related keys that might cause issues
        # Fix for IndexError in cache_position handling (transformers 4.57.3 bug)
        # Remove cache-related keys that cause issues with Qwen2-VL
        clean_inputs = {k: v for k, v in inputs.items() 
                       if k not in ["past_key_values", "cache_position", "position_ids"]}
        
        outputs = model.generate(
            **clean_inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            use_cache=False,  # Disable KV cache to avoid IndexError with empty cache_position
        )

    # Decode only the newly generated tokens
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    reply = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return reply, orig_w, orig_h


def parse_xy_from_string(text: str) -> tuple[int, int]:
    """
    Extract (x, y) from a string like '(123, 456)'.
    """
    m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if not m:
        raise ValueError(f"Could not parse coordinates from: {text!r}")
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x < 1000 and 0 <= y < 1000):
        raise ValueError(f"Coordinates out of [0,1000) range: {(x, y)} from {text!r}")
    return x, y


def scale_to_pixels(x_1000: int, y_1000: int, width: int, height: int) -> tuple[int, int]:
    """
    Map model coordinates in [0,1000) to pixel coordinates of the original image.
    """
    x_px = int(x_1000 / 1000 * width)
    y_px = int(y_1000 / 1000 * height)
    x_px = max(0, min(width - 1, x_px))
    y_px = max(0, min(height - 1, y_px))
    return x_px, y_px


def call_grounding_model(
    image_path: str,
    query: str,
    model_key: str,
) -> tuple[int, int, str]:
    """
    Run grounding with the selected model.

    model_key must be one of MODEL_CONFIGS keys, e.g.:
        - "uground-2b"
        - "qwen2-vl-2b"
    """
    if model_key not in PROCESSORS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(PROCESSORS.keys())}")

    processor = PROCESSORS[model_key]
    model = MODELS[model_key]

    # 1. 根据传入的是路径还是 data URL 来获取 PIL.Image
    if image_path.startswith("data:image"):
        # data:image/png;base64,xxxx 这种
        header, encoded = image_path.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    else:
        # 还是老的本地路径/文件名逻辑
        img = Image.open(image_path).convert("RGB")

    # Reuse the shared raw call helper
    reply, orig_w, orig_h = call_ui_tars_raw(
        image=img,
        query=query,
        processor=processor,
        model=model,
    )

    x_1000, y_1000 = parse_xy_from_string(reply)
    x_px, y_px = scale_to_pixels(x_1000, y_1000, orig_w, orig_h)
    return x_px, y_px, reply



# ============
# 3. API
# ============

class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: List[ContentPart]


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: float = 0.0
    max_tokens: int = 128


class ChatMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]


app = FastAPI()


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    """
    一个最小 OpenAI ChatCompletion 兼容接口：
    - messages[...].content 里：
        - {"type": "text", "text": "...描述..."}
        - {"type": "image_url", "image_url": {"url": "test.png"}}
    - image_url.url 可以是本地路径，比如 "test.png" 或 "file://test.png"
    """
    user_msg = None
    for m in reversed(req.messages):
        if m.role == "user":
            user_msg = m
            break
    if user_msg is None:
        raise HTTPException(status_code=400, detail="No user message found.")

    description = None
    image_path = None

    for part in user_msg.content:
        if part.type == "text" and part.text:
            description = part.text
        elif part.type == "image_url" and part.image_url:
            url = part.image_url.url
            if url.startswith("file://"):
                url = url[len("file://"):]
            image_path = url

    if not description or not image_path:
        raise HTTPException(
            status_code=400,
            detail="Need both text description and image_url in user content.",
        )
    # Decide which underlying model to use
    # If req.model is None, fall back to DEFAULT_MODEL_KEY
    model_key = req.model or DEFAULT_MODEL_KEY

    # Run blocking model inference in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    x_px, y_px, raw = await loop.run_in_executor(
        _inference_executor,
        call_grounding_model,
        image_path,
        description,
        model_key
    )

    msg = ChatMessage(
        role="assistant",
        content=raw,
    )
    choice = Choice(index=0, message=msg, finish_reason="stop")

    resp = ChatResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        object="chat.completion",
        created=int(time.time()),
        model=req.model or model_key,
        choices=[choice],
    )
    return resp
