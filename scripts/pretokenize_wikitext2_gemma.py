#!/usr/bin/env python3
"""
Offline tokenize WikiText-2 with the HF Gemma tokenizer into a single int32 stream.
Writes <output_dir>/wt2_gemma_tokens.bin and <output_dir>/meta.json.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer


def read_lines(path: Path, keep_blank: bool) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not keep_blank and not line.strip():
                continue
            lines.append(line.rstrip("\n"))
    return lines


def pack_tokens(lines: List[str],
                tokenizer,
                insert_eos_between_lines: bool) -> List[int]:
    ids: List[int] = []
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer is missing eos_token_id")
    for line in lines:
        encoded = tokenizer.encode(line, add_special_tokens=False)
        ids.extend(int(x) for x in encoded)
        if insert_eos_between_lines:
            ids.append(int(eos_id))
    if not ids or ids[-1] != eos_id:
        ids.append(int(eos_id))
    return ids


def resolve_pad_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretokenize WikiText-2 with Gemma tokenizer.")
    parser.add_argument("--model_dir", default="gemma-3-270m", help="Path to Gemma tokenizer files.")
    parser.add_argument("--data_dir", default="data/wikitext2/wikitext-2-raw", help="WikiText-2 raw folder.")
    parser.add_argument("--output_dir", default="data/wikitext2/pretokenized_gemma", help="Where to write bin/meta.")
    parser.add_argument("--output_name", default="wt2_gemma_tokens.bin", help="Token stream filename.")
    parser.add_argument("--preview", type=int, default=50, help="Print first N tokens for sanity check.")
    parser.add_argument("--no_insert_eos_between_lines", dest="insert_eos_between_lines",
                        action="store_false", help="Do not add EOS between lines.")
    parser.set_defaults(insert_eos_between_lines=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pad_id = resolve_pad_id(tokenizer)

    split_files: Dict[str, Path] = {
        "train": data_dir / "wiki.train.raw",
        "valid": data_dir / "wiki.valid.raw",
        "test": data_dir / "wiki.test.raw",
    }

    all_tokens: List[int] = []
    splits: Dict[str, Dict[str, int]] = {}
    offset = 0
    for split, path in split_files.items():
        if not path.exists():
            continue
        lines = read_lines(path, args.insert_eos_between_lines)
        ids = pack_tokens(lines, tokenizer, args.insert_eos_between_lines)
        splits[split] = {
            "offset": offset,
            "length": len(ids),
            "file": str(path),
        }
        all_tokens.extend(ids)
        offset += len(ids)

    if not all_tokens:
        raise RuntimeError("No tokens produced; check data_dir/model_dir paths.")

    token_array = np.asarray(all_tokens, dtype=np.int32)
    bin_path = output_dir / args.output_name
    token_array.tofile(bin_path)

    meta = {
        "total_tokens": int(token_array.size),
        "dtype": "int32",
        "endianness": "little",
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1,
        "pad_token_id": int(pad_id),
        "bos_token_id": int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else -1,
        "unk_token_id": int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else -1,
        "vocab_size": int(tokenizer.vocab_size),
        "insert_eos_between_lines": bool(args.insert_eos_between_lines),
        "splits": splits,
        "token_stream_path": str(bin_path),
        "source": {
            "model_dir": str(model_dir),
            "data_dir": str(data_dir),
        },
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if args.preview and args.preview > 0:
        head = all_tokens[:args.preview]
        print(f"First {len(head)} tokens: {head}")
        print(f"Total tokens written: {token_array.size}")
        for name, info in splits.items():
            print(f"{name}: offset={info['offset']} length={info['length']}")


if __name__ == "__main__":
    main()
