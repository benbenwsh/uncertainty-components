"""Read the first N examples from a generations HDF5 file and write them to JSON."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

EMBEDDING_KEYS = {
    "emb_sec_last_token",
    "emb_tok_bef_gen",
    "all_embeddings",
    "all_attn_embeddings",
    "all_mlp_embeddings",
    "all_q_embeddings",
    "all_k_embeddings",
    "all_v_embeddings",
    "all_o_embeddings",
    "all_concat_embeddings",
}


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_scalar(value.item())
    return value


def _summarize_dataset(node: h5py.Dataset):
    data = node[()]
    decoded = _decode_scalar(data)
    if isinstance(decoded, np.ndarray):
        if decoded.dtype.kind in {"S", "U", "O"}:
            items = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in decoded.tolist()
            ]
            return items
        return {
            "__summary__": f"ndarray shape={list(decoded.shape)} dtype={str(decoded.dtype)}",
            "preview": decoded.reshape(-1)[:10].tolist(),
        }
    return decoded


def _read_h5_node(node, *, summarize_arrays=False):
    if isinstance(node, h5py.Dataset):
        if summarize_arrays:
            return _summarize_dataset(node)
        return _summarize_dataset(node)

    if isinstance(node, h5py.Group):
        node_type = node.attrs.get("__type__")
        if isinstance(node_type, bytes):
            node_type = node_type.decode("utf-8")
        if node_type == "none":
            return None
        if node_type in {"list", "tuple"}:
            length = int(node.attrs.get("__len__", 0))
            if summarize_arrays and length > 3:
                return {
                    "__summary__": f"{node_type} length {length}",
                    "first_items": [
                        _read_h5_node(node[str(i)], summarize_arrays=summarize_arrays)
                        for i in range(min(3, length))
                    ],
                }
            return [
                _read_h5_node(node[str(i)], summarize_arrays=summarize_arrays)
                for i in range(length)
            ]
        out = {}
        for k, v in node.items():
            summarize = summarize_arrays or k in EMBEDDING_KEYS
            out[k] = _read_h5_node(v, summarize_arrays=summarize)
        return out

    raise TypeError(f"Unsupported HDF5 node type: {type(node)}")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def main():
    parser = argparse.ArgumentParser(description="Export first N HDF5 examples to JSON.")
    parser.add_argument(
        "--input",
        default="ans_gen/generated_answers/2/train_generations.h5",
        help="Path to train_generations.h5 or validation_generations.h5",
    )
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <input_dir>/first_<N>_examples.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"first_{args.num_examples}_examples.json"
    )

    examples = {}
    with h5py.File(input_path, "r") as h5_file:
        ex_group = h5_file["examples"]
        for example_id in list(ex_group.keys())[: args.num_examples]:
            examples[example_id] = _read_h5_node(ex_group[example_id])

            ml = examples[example_id].get("most_likely_answer", {})
            if isinstance(ml, dict) and ml.get("decoded_tokens"):
                decoded_tokens = ml["decoded_tokens"]
                examples[example_id]["joined_decoded_tokens"] = "".join(decoded_tokens)
                print(f"\n=== {example_id} ===")
                print("decoded_tokens:")
                print(json.dumps(decoded_tokens, indent=2, ensure_ascii=False))
                print(f"joined: {examples[example_id]['joined_decoded_tokens']!r}")

    payload = _to_jsonable(
        {
            "source": str(input_path.resolve()),
            "num_examples": len(examples),
            "examples": examples,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
