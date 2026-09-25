#!/usr/bin/env python3
"""Dependency-free GGUF metadata reader used to pull a model's OWN embedded jinja chat template
(tokenizer.chat_template) straight out of the .gguf file header, without loading any tensor data
and without any GPU/llama-server involvement.

Used to derive the *.orig.jinja reference copies in this directory (the unmodified template each
model ships with) before hand-patching them into the *.jinja files that profiles.json points
--chat-template-file at. Re-run this if a model file is ever replaced, to re-diff the new upstream
template against the local patch.

Usage:
    python3 extract_template.py <model.gguf>                       # list all metadata keys
    python3 extract_template.py <model.gguf> tokenizer.chat_template > model.orig.jinja
"""
import struct
import sys

GGUF_MAGIC = 0x46554747

(TYPE_UINT8, TYPE_INT8, TYPE_UINT16, TYPE_INT16, TYPE_UINT32, TYPE_INT32, TYPE_FLOAT32,
 TYPE_BOOL, TYPE_STRING, TYPE_ARRAY, TYPE_UINT64, TYPE_INT64, TYPE_FLOAT64) = range(13)

SCALAR_FMT = {
    TYPE_UINT8: ("B", 1), TYPE_INT8: ("b", 1), TYPE_UINT16: ("H", 2), TYPE_INT16: ("h", 2),
    TYPE_UINT32: ("I", 4), TYPE_INT32: ("i", 4), TYPE_FLOAT32: ("f", 4), TYPE_BOOL: ("?", 1),
    TYPE_UINT64: ("Q", 8), TYPE_INT64: ("q", 8), TYPE_FLOAT64: ("d", 8),
}


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected EOF reading %d bytes" % n)
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def gstr(self):
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == TYPE_STRING:
            return self.gstr()
        if vtype == TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            return [self.value(elem_type) for _ in range(n)]
        if vtype in SCALAR_FMT:
            fmt, size = SCALAR_FMT[vtype]
            return struct.unpack("<" + fmt, self.read(size))[0]
        raise ValueError("unknown GGUF value type %d" % vtype)


def read_metadata(path):
    """Read only the GGUF header + metadata KV section (never the tensor data that follows it)."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        magic = r.u32()
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file (magic=%08x)" % magic)
        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()
        meta = {}
        for _ in range(kv_count):
            key = r.gstr()
            vtype = r.u32()
            meta[key] = r.value(vtype)
        return {"version": version, "tensor_count": tensor_count, "metadata": meta}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    key = argv[2] if len(argv) > 2 else None
    info = read_metadata(path)
    if key:
        v = info["metadata"].get(key)
        if v is None:
            print("KEY NOT FOUND: %s" % key, file=sys.stderr)
            print("available keys with 'template' or 'chat':", file=sys.stderr)
            for k in info["metadata"]:
                if "template" in k.lower() or "chat" in k.lower():
                    print("  " + k, file=sys.stderr)
            return 1
        sys.stdout.write(v if isinstance(v, str) else repr(v))
        return 0
    for k, v in info["metadata"].items():
        if isinstance(v, str) and len(v) > 200:
            print("%s: <string, %d chars>" % (k, len(v)))
        elif isinstance(v, list) and len(v) > 20:
            print("%s: <array, %d items>" % (k, len(v)))
        else:
            print("%s: %r" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
