import argparse
import datetime
import json
import logging
import os
import sys
import time
import re

from types import SimpleNamespace

from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.agents.grounding import OSWorldACI


def sanitize_for_path(name: str) -> str:
    """Replace any char that is not letter, digit, underscore, dot, or dash."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main():
    # -------------------------------------------------------------------------
    # Config (edit here as needed)
    # -------------------------------------------------------------------------
    args = SimpleNamespace(
        # environment config
        path_to_vm=None,
        provider_name="vmware",          # TODO: change to your actual provider
        headless=True,
        action_space="pyautogui",
        observation_type="screenshot_a11y_tree",
        screen_width=1920,
        screen_height=1080,
        sleep_after_execution=3.0,
        max_steps=10,                     # small for debugging

        # agent config
        test_config_base_dir="evaluation_examples",

        # main LM config (e.g., Ollama / Qwen / etc.)
        model="gemma3:27b",
        temperature=0.2,
        model_provider="openai",         # Agent-S uses OpenAI-style client
        model_url="http://localhost:11434/v1",
        model_api_key="ollama",
        model_temperature=0.2,

        # grounding model config (reuse same endpoint/model for now)
        ground_provider="openai",
        ground_url="http://localhost:1234/v1",
        ground_api_key="lm-studio",
        ground_model="ui-tars-7b-dpo@q4_k_s",
        grounding_width=1000,
        grounding_height=1000,

        # example config
        domain="all",
        test_all_meta_path="evaluation_examples/test_all.json",

        # logging / results
        result_dir="./results",
    )

    # -------------------------------------------------------------------------
    # Build engine params for main LM and grounding LM
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # OSWorld environment (Docker / VMware / etc. provider)
    # -------------------------------------------------------------------------
    env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type
        in ["a11y_tree", "screenshot_a11y_tree", "som"],
        enable_proxy=False,   # TURNED OFF for now
    )

    # -------------------------------------------------------------------------
    # Agent-S grounding agent + Agent-S3 main agent
    # -------------------------------------------------------------------------
    grounding_agent = OSWorldACI(
        env=env,
        platform="linux",
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=args.screen_width,
        height=args.screen_height,
    )

    agent = AgentS3(
        engine_params,
        grounding_agent,
        platform="linux",
    )

    safe_model_name = sanitize_for_path(args.model)

    # -------------------------------------------------------------------------
    # Load test metadata (test_all.json) and pick examples
    # -------------------------------------------------------------------------
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    # Take the first domain (or override args.domain manually if you like)
    domain = next(iter(test_all_meta.keys()))
    # First 10 tasks in this domain (change 10 to 1 while debugging if desired)
    example_ids = test_all_meta[domain][8:11]

    print("Domain:", domain)
    print("Example IDs:", example_ids)

    results = []  # to store per-task stats

    try:
        for i, example_id in enumerate(example_ids):
            # Skip example 7 for now
            if i == 7:
                continue
            print("\n==============================")
            print(f"Running example: {domain}/{example_id}")

            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            print("Config file path:", config_file)

            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            print("Instruction:", example["instruction"])

            # Result dir for this example
            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                safe_model_name,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)
            print("Result dir:", example_result_dir)

            # For this single run, collect scores locally
            scores: list[float] = []

            start = time.perf_counter()
            lib_run_single.run_single_example(
                agent,
                env,
                example,
                args.max_steps,
                example["instruction"],
                args,
                example_result_dir,
                scores,
            )
            end = time.perf_counter()
            latency = end - start

            print("Scores list:", scores)

            # Assume the last entry is the final score (adjust if your API differs)
            final_score = scores[-1] if scores else None

            # Define "success" for accuracy — adjust rule if needed
            # Common pattern: score 1.0 = success, 0.0 = fail
            success = None
            if final_score is not None:
                if final_score in (0, 1):
                    success = bool(final_score)
                else:
                    # Fallback: treat positive score as success
                    success = final_score > 0

            results.append(
                {
                    "domain": domain,
                    "example_id": example_id,
                    "latency": latency,
                    "final_score": final_score,
                    "success": success,
                }
            )

            print(f"Latency: {latency:.3f} seconds")
            print(f"Final score: {final_score}")
            print(f"Success: {success}")

    finally:
        # Always close environment
        env.close()

    # -------------------------------------------------------------------------
    # Summary over tasks
    # -------------------------------------------------------------------------
    valid_success = [r["success"] for r in results if r["success"] is not None]
    accuracy = sum(valid_success) / len(valid_success) if valid_success else 0.0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0.0

    print("\n===== Summary over tasks =====")
    for r in results:
        print(
            f"{r['domain']}/{r['example_id']}: "
            f"score={r['final_score']}, success={r['success']}, latency={r['latency']:.3f}s"
        )

    print(f"\nOverall accuracy: {accuracy * 100:.1f}%")
    print(f"Average latency: {avg_latency:.3f} seconds")


if __name__ == "__main__":
    main()
