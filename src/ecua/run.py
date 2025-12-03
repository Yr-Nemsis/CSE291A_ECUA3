import argparse
import datetime
import json
import logging
import os
import sys
import atexit

from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.agents.grounding import OSWorldACI

from types import SimpleNamespace


args = SimpleNamespace(
    # environment config
    path_to_vm=None,
    provider_name="docker",          # <--- not vmware
    headless=True,
    action_space="pyautogui",
    # observation_type="a11y_tree",
    # observation_type="screenshot_a11y_tree",
    observation_type="screenshot",
    screen_width=1920,
    screen_height=1080,
    # screen_width=1280,
    # screen_height=720,
    sleep_after_execution=2.5,
    max_steps=15,                     # small for debugging

    # agent config
    max_trajectory_length=1,
    test_config_base_dir="evaluation_examples",

    # main LM config (Ollama llama3.2:3b) now testing with llama3.1:8b
    # deepseek-llm:7b(too slow) llama3.2:3b(bad reasoning) llama3.1:8b(best so far)
    model="llama3.1:8b",
    temperature=0.2,
    model_provider="openai",         # Agent-S uses OpenAI-style client
    model_url="http://localhost:11434/v1",
    model_api_key="ollama",
    model_temperature=0.2,

    # grounding model config (reuse same endpoint/model for now) testing with UI-TARS
    # UITARS 2B from hugging face(encoding issue?) InternVL3_5:4b(wrong detection) UITARS 1.5B(16gb, too slow)
    ground_provider="openai",
    # ground_url="http://localhost:11434/v1",
    ground_url="http://localhost:8000/v1", # new endpoint for UI-TARS
    ground_api_key="ollama",
    ground_model="ui-tars-7b",
    # grounding_width=1920,
    # grounding_height=1080,
    grounding_width=1000,
    grounding_height=1000,

    # example config
    domain="all",
    test_all_meta_path="evaluation_examples/test_all.json",

    # logging / results
    result_dir="./results",
)

# Build engine params
engine_params = {
    "engine_type": args.model_provider,
    "model": args.model,
    "base_url": getattr(args, "model_url", ""),
    "api_key": getattr(args, "model_api_key", ""),
    "temperature": getattr(args, "model_temperature", None),
}

engine_params_for_grounding = {
    "engine_type": args.ground_provider,
    "model": args.ground_model,
    "base_url": getattr(args, "ground_url", ""),
    "api_key": getattr(args, "ground_api_key", ""),
    "grounding_width": args.grounding_width,
    "grounding_height": args.grounding_height,
}

# OSWorld environment (Docker provider)
env = DesktopEnv(
    provider_name=args.provider_name,
    path_to_vm=args.path_to_vm,
    action_space=args.action_space,
    screen_size=(args.screen_width, args.screen_height),
    headless=args.headless,
    os_type="Ubuntu",
    require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
    enable_proxy=False,   # <--- TURNED OFF for now
)

atexit.register(env.close)

# Agent-S grounding agent
grounding_agent = OSWorldACI(
    env=env,
    platform="linux",
    engine_params_for_generation=engine_params,
    engine_params_for_grounding=engine_params_for_grounding,
    width=args.screen_width,
    height=args.screen_height,
)

# Agent-S3 main agent
agent = AgentS3(
    engine_params,
    grounding_agent,
    platform="linux",
)

with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
    test_all_meta = json.load(f)

# For a first run, choose the first domain and the second example ID
domain = next(iter(test_all_meta.keys()))
example_id = test_all_meta[domain][0]

print("Domain:", domain)
print("Example ID:", example_id)

config_file = os.path.join(
    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
)
print("Config file path:", config_file)

with open(config_file, "r", encoding="utf-8") as f:
    example = json.load(f)

print("Instruction:", example["instruction"])

scores: list[float] = []
example_result_dir = os.path.join(
    args.result_dir,
    args.action_space,
    args.observation_type,
    args.model,
    domain,
    example_id,
)
os.makedirs(example_result_dir, exist_ok=True)

print("Result dir:", example_result_dir)

system_prefix = ''
"""
You are an agent controlling a Linux desktop environment via mouse and keyboard.
Your primary application is **Google Chrome**, and almost all tasks should be solved
by operating inside Chrome (the web browser) unless the user explicitly requires
other applications.

You receive visual observations (screenshots) of the desktop and must decide
what actions to take. Your goal is to complete the user’s instruction reliably,
with as few steps and mistakes as possible.

-------------------------
GLOBAL PRINCIPLES
-------------------------

1. **Work inside Chrome whenever possible**
   - If the task involves the web, URLs, websites, online search, accounts, or web apps,
     you MUST use Google Chrome as the main tool.
   - Do NOT open the OS “Settings” application or system-level tools unless the user
     explicitly requests system configuration (e.g., change system language).
   - Do NOT use other browsers (Firefox, etc.) unless explicitly instructed.

2. **Make sure Chrome is open and focused**
   - If Chrome is not visible, open it using the desktop UI (e.g., from the taskbar,
     application menu, or an existing shortcut).
   - If multiple windows are open, select the Chrome window that is most relevant
     to the user’s request (e.g., where the active tab is visible).
   - Avoid opening many duplicate Chrome windows; reuse an existing window if possible.

3. **Use the Chrome UI correctly**
   Typical Chrome operations include:
   - Opening a new tab (e.g., “+” button near the top or keyboard shortcut).
   - Entering a URL or query in the address bar.
   - Using the back/forward/refresh buttons.
   - Opening the menu (three dots “⋮” in the top-right corner) to access:
     - Settings
     - History
     - Downloads
     - Extensions
   - Switching tabs by clicking on the tab strip.
   - Scrolling pages to find relevant content.
   - Clicking buttons, links, text fields, dropdowns, and other UI elements on web pages.

4. **Rely on visual evidence from the screenshot**
   - Always examine the screenshot carefully before clicking.
   - Identify elements by their visible text, icons, position, and context.
   - Do NOT guess coordinates. Every click should have a clear visual target.
   - If you cannot find the desired element, scroll or adjust the view instead of
     randomly clicking.

5. **Be deliberate and minimize unnecessary actions**
   - Do NOT open random applications or windows.
   - Do NOT type into irrelevant text fields.
   - Avoid repeated clicks on the same location unless needed (e.g., opening a menu).
   - Prefer structured navigation (menus, well-labeled buttons) over arbitrary clicking.

6. **Respect forms, inputs, and user data**
   - When filling forms, type carefully and only in appropriate fields.
   - Do NOT modify settings or data unrelated to the user’s request.
   - Avoid destructive actions (e.g., deleting data, logging out, uninstalling extensions)
     unless explicitly required by the task.

7. **Verification and task completion**
   - Whenever reasonable, verify that your actions had the intended effect.
     Examples:
       - After navigating to a website, confirm that the URL or page title matches
         what the user asked for.
       - After changing a setting in Chrome, revisit the relevant UI or perform a
         small test (e.g., reload a page, open a new tab) to confirm the change.
   - Only consider the task complete when there is clear visual evidence that the
     user’s goal has been achieved.

8. **Handle unexpected states robustly**
   - If a popup appears (e.g., cookie banner, permission dialog), handle it in a
     reasonable way so you can continue the main task.
   - If an error page appears (network error, 404, etc.), try to diagnose and
     recover (e.g., check the URL, reload, go back, or try a different link).
   - If you are stuck or the UI does not match your expectations, take a small
     number of exploratory steps (scroll, move the mouse, open menu) but avoid
     chaotic clicking.

-------------------------
BEHAVIORAL RULES
-------------------------

- Prefer **Chrome menu + Settings** for browser configuration tasks (privacy, search,
  startup pages, extensions, etc.), not OS-level settings.
- For web search, always use the address bar or the main content search box,
  not random page elements.
- Keep the desktop as clean as possible: close irrelevant windows or tabs
  when they interfere with the task.
- Never assume a change succeeded without checking the UI; when in doubt,
  perform a small verification step in Chrome.
- Give only one grounding action per step; do not chain multiple clicks or types

You must act like a careful, detail-oriented human user who is proficient with
Google Chrome and the web. Your actions should always be explainable by what you
see on the screen.

"""

lib_run_single.run_single_example(
    agent,
    env,
    example,
    args.max_steps,
    system_prefix + example["instruction"],
    args,
    example_result_dir,
    scores,
)

print("Scores list:", scores)

env.close()