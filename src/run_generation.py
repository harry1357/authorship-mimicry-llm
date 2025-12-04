# src/run_generation.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from generation_config import PROMPTS_DIR, GENERATED_DIR, DEFAULT_LLM_KEY
from llm_client import get_llm_client, LLMRequest, LLMResponse


def load_prompts(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-run",
        type=int,
        choices=[1, 2],
        default=1,
        help="Which full run to generate for (1 or 2).",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default=DEFAULT_LLM_KEY,
        help="LLM key to use (e.g. 'gpt-5.1' or 'mock').",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional explicit prompts file path.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional explicit output file path.",
    )
    args = parser.parse_args()

    full_run = args.full_run
    llm_key = args.llm_key

    if args.prompts_file is None:
        prompts_path = PROMPTS_DIR / f"generation_prompts_fullrun{full_run}.jsonl"
    else:
        prompts_path = Path(args.prompts_file)

    if args.output_file is None:
        out_dir = GENERATED_DIR / llm_key
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"generations_fullrun{full_run}.jsonl"
    else:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run_generation] full_run={full_run} llm={llm_key}")
    print(f"[run_generation] prompts: {prompts_path}")
    print(f"[run_generation] output: {output_path}")

    client = get_llm_client(llm_key)

    num = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for prompt_rec in load_prompts(prompts_path):
            prompt_id = prompt_rec["prompt_id"]
            author_id = prompt_rec["author_id"]
            full_run_rec = int(prompt_rec.get("full_run", full_run))
            prompt_index = int(prompt_rec.get("prompt_index", 1))

            req = LLMRequest(
                prompt_id=prompt_id,
                author_id=author_id,
                run_id=full_run_rec,
                prompt_text=prompt_rec["prompt_text"],
                max_tokens=prompt_rec.get("max_tokens", 1024),
                temperature=prompt_rec.get("temperature", 0.8),
                seed=prompt_rec.get("seed"),
                metadata=prompt_rec.get("metadata"),
            )

            print(
                f"[run_generation] author={author_id} prompt_id={prompt_id} "
                f"full_run={full_run_rec} p={prompt_index}"
            )

            resp: LLMResponse = client.generate(req)

            record: Dict[str, Any] = {
                "generation_id": f"{prompt_id}__{llm_key}",
                "prompt_id": resp.prompt_id,
                "author_id": resp.author_id,
                "full_run": full_run_rec,
                "prompt_index": prompt_index,
                "llm_key": resp.llm_key,
                "generated_text": resp.generated_text,
                "usage": resp.usage,
                "prompt_text": req.prompt_text,
                "prompt_metadata": req.metadata,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num += 1

    print(f"[run_generation] Wrote {num} generations to {output_path}")


if __name__ == "__main__":
    main()