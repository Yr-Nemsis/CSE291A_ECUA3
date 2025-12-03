import re
import time
import uuid
from typing import List, Optional
import base64
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq


MODEL_ID = "ByteDance-Seed/UI-TARS-7B-DPO"

# ==== 1. Load processor + model on CPU only ====
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # smaller than float32, fine on CPU
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()


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


def call_ui_tars_raw(image_path: str, query: str) -> tuple[str, int, int]:
    """
    Send one screenshot + grounding query to UI-TARS, return:
    (raw_text_response, orig_width, orig_height).
    """
    # Load original image
    img = Image.open(image_path).convert("RGB")
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

    # Move to CPU (model is on CPU)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
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


def call_grounding_model(image_path: str, query: str) -> tuple[int, int, str]:
    # 1. 根据传入的是路径还是 data URL 来获取 PIL.Image
    if image_path.startswith("data:image"):
        # data:image/png;base64,xxxx 这种
        header, encoded = image_path.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    else:
        # 还是老的本地路径/文件名逻辑
        img = Image.open(image_path).convert("RGB")

    orig_w, orig_h = img.size

    # 2. 后面保持你原来的逻辑（只把原来的 Image.open(image_path) 换成 img）
    messages = build_uground_messages(query)

    chat_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[chat_prompt],
        images=[img],
        return_tensors="pt",
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    reply = processor.decode(gen_ids, skip_special_tokens=True).strip()

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
def chat_completions(req: ChatRequest):
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

    x_px, y_px, raw = call_grounding_model(image_path, description)

    msg = ChatMessage(
        role="assistant",
        content=raw,
    )
    choice = Choice(index=0, message=msg, finish_reason="stop")

    resp = ChatResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        object="chat.completion",
        created=int(time.time()),
        model=req.model or MODEL_ID,
        choices=[choice],
    )
    return resp
