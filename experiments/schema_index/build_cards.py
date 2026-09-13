#!/usr/bin/env python3
"""Build the retrieval corpus ("field cards") out of Coddy's config.schema.json.

The point of this script is to measure, not to ship: how many addressable config
paths there are, how much text a card costs, and how much of the schema a small
model would have to look at if we gave it everything instead of retrieving.

Usage: python3 build_cards.py [path/to/config.schema.json] [-o cards.jsonl]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFAULT_SCHEMA = pathlib.Path(
    "/storage/Repository/coddy/coddy-agent/internal/config/config.schema.json"
)


def resolve(node, root):
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 10:
        ref = node["$ref"]
        seen += 1
        if not ref.startswith("#/"):
            return node
        cur = root
        for part in ref[2:].split("/"):
            cur = cur.get(part, {})
        node = cur
    return node


def type_of(node):
    t = node.get("type")
    if isinstance(t, list):
        return "|".join(x for x in t if x != "null") or "null"
    if t:
        return t
    for comb in ("oneOf", "anyOf"):
        if comb in node:
            parts = [type_of(x) for x in node[comb]]
            return "|".join(sorted({p for p in parts if p and p != "null"}))
    if "properties" in node:
        return "object"
    if "enum" in node:
        return "enum"
    return "any"


def walk(node, root, path, out, depth=0):
    node = resolve(node, root)
    if not isinstance(node, dict) or depth > 12:
        return
    props = node.get("properties")
    if props:
        for key, sub in props.items():
            walk(sub, root, path + [key], out, depth + 1)
        # an object with properties is still addressable (delete <path>)
        if path:
            out.append(card(node, path, "object"))
        return
    if node.get("type") == "array" or "items" in node:
        items = resolve(node.get("items", {}), root)
        if items.get("properties"):
            # a list of objects: address entries with name[key=value].<field>
            for key, sub in items["properties"].items():
                walk(sub, root, path + ["[]", key], out, depth + 1)
        out.append(card(node, path, "array<%s>" % type_of(items)))
        return
    for comb in ("oneOf", "anyOf", "allOf"):
        if comb in node:
            for sub in node[comb]:
                sub = resolve(sub, root)
                if sub.get("properties") or sub.get("items"):
                    walk(sub, root, path, out, depth + 1)
                    return
    if path:
        out.append(card(node, path, type_of(node)))


def card(node, path, kind):
    c = {
        "path": ".".join(path),
        "section": path[0],
        "type": kind,
        "description": (node.get("description") or "").strip(),
    }
    if "enum" in node:
        c["enum"] = node["enum"]
    if "default" in node:
        c["default"] = node["default"]
    for k in ("minimum", "maximum", "pattern"):
        if k in node:
            c[k] = node[k]
    return c


def render(c):
    """One card as the model would see it."""
    head = "%s : %s" % (c["path"], c["type"])
    bits = []
    if "enum" in c:
        bits.append("enum=" + ",".join(str(x) for x in c["enum"]))
    if "default" in c:
        bits.append("default=" + json.dumps(c["default"], ensure_ascii=False))
    if bits:
        head += "  (" + "; ".join(bits) + ")"
    desc = c["description"]
    return head + ("\n  " + desc if desc else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schema", nargs="?", default=str(DEFAULT_SCHEMA))
    ap.add_argument("-o", "--out", default="cards.jsonl")
    args = ap.parse_args()

    root = json.loads(pathlib.Path(args.schema).read_text(encoding="utf-8"))
    cards: list[dict] = []
    walk(root, root, [], cards)

    uniq = {}
    for c in cards:
        uniq.setdefault(c["path"], c)
    cards = sorted(uniq.values(), key=lambda c: c["path"])

    with open(args.out, "w", encoding="utf-8") as fh:
        for c in cards:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    text = "\n".join(render(c) for c in cards)
    chars = len(text)
    # 1 token ~ 3.6 chars for English technical text; Cyrillic is worse, but the
    # schema itself is English.
    approx_tokens = round(chars / 3.6)
    leaves = [c for c in cards if c["type"] not in ("object",)]
    with_desc = [c for c in cards if c["description"]]
    with_enum = [c for c in cards if "enum" in c]
    by_section: dict[str, int] = {}
    for c in cards:
        by_section[c["section"]] = by_section.get(c["section"], 0) + 1

    print("cards total      :", len(cards))
    print("  addressable    :", len(leaves), "(non-object)")
    print("  with desc      :", len(with_desc))
    print("  with enum      :", len(with_enum))
    print("full corpus      : %d chars, ~%d tokens" % (chars, approx_tokens))
    print("avg card         : %.0f chars, ~%.0f tokens" % (chars / len(cards), approx_tokens / len(cards)))
    print("sections         :", len(by_section))
    for s, n in sorted(by_section.items(), key=lambda kv: -kv[1]):
        print("   %-14s %3d" % (s, n))
    print("\nsample:\n")
    for c in cards[:3]:
        print(render(c))
        print()


if __name__ == "__main__":
    sys.exit(main())
