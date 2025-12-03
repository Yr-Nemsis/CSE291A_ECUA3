import re
import torch
from PIL import Image
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
    Build chat messages in the format expected by Qwen2-VL-style processors,
    with a local image (provided separately to `processor`).
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


def call_ui_tars_raw(image_path: str, query: str) -> tuple[str, int, int, int, int]:
    """
    Send one screenshot + grounding query to UI-TARS, return:
    (raw_text_response, orig_width, orig_height, proc_width, proc_height).


    We:
    - load the original image
    - compute smart_resize() dimensions
    - resize to (proc_width, proc_height) for the model
    - prompt in Thought/Action format with click(start_box='(x,y)')
    """
    # Load original image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size


    # # Use smart_resize as the actual model input size
    # proc_h, proc_w = smart_resize(orig_h, orig_w)
    # img_proc = orig_img.resize((proc_w, proc_h), Image.LANCZOS)


    # Prompt in UI-TARS-style format
    # (COMPUTER_USE-style Thought + Action with click(start_box='(x,y)'))
    prompt = build_uground_messages(query)


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # the actual image is passed via `images=` below
                {"type": "text", "text": prompt},
            ],
        },
    ]


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


    print(reply)


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
        # Not strictly necessary, but good sanity check
        raise ValueError(f"Coordinates out of [0,1000) range: {(x, y)} from {text!r}")
    return x, y




def scale_to_pixels(x_1000: int, y_1000: int, width: int, height: int) -> tuple[int, int]:
    """
    Map model coordinates in [0,1000) to pixel coordinates of the original image.
    """
    x_px = int(x_1000 / 1000 * width)
    y_px = int(y_1000 / 1000 * height)
    # Clamp just in case of boundary issues
    x_px = max(0, min(width - 1, x_px))
    y_px = max(0, min(height - 1, y_px))
    return x_px, y_px


def call_grounding_model(image_path: str, query: str) -> tuple[int, int, str]:
    raw, orig_w, orig_h = call_ui_tars_raw(image_path, query)
        # 6) Parse (x, y) in [0,1000)
    x_1000, y_1000 = parse_xy_from_string(raw)


    # 7) Map to pixel coordinates
    x_px, y_px = scale_to_pixels(x_1000, y_1000, orig_w, orig_h)


    # x, y = response_to_click_xy(raw, orig_w, orig_h)
    return x_px, y_px, raw


import matplotlib.pyplot as plt


img_path = "test2.png"
query = "the 'Customize Chrome' button at the bottom right corner of the window"


x, y, raw = call_grounding_model(img_path, query)
print("Raw model output:\n", raw)
print("Predicted click in original image coords:", x, y)


img = Image.open(img_path)
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.scatter([x], [y], s=50)
plt.title("UI-TARS predicted click")
plt.axis("off")
plt.savefig("coordinate_process_image_som.png", dpi=300, bbox_inches="tight")


