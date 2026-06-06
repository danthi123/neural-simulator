"""Download CIFAR-10 (32x32 natural object photos) for real-object grounding (option 2). Owner explicitly authorized
the download. Source = the CANONICAL Toronto URL (trusted by construction); extraction uses the py3.12+ safe filter;
data/ is gitignored (not committed). The runner loads the resulting official pickle batches (trusted source)."""
import os
import shutil
import tarfile
import urllib.request

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEST = "data/cifar10"
os.makedirs(DEST, exist_ok=True)

# ensure data/ is gitignored (the 163 MB dataset must not be committed)
gi = ".gitignore"
have = open(gi).read() if os.path.exists(gi) else ""
if "data/" not in have:
    with open(gi, "a") as f:
        f.write("\n# large datasets (not committed)\ndata/\n")

batch1 = os.path.join(DEST, "cifar-10-batches-py", "data_batch_1")
if os.path.exists(batch1):
    print("CIFAR-10 already present:", batch1, flush=True)
    raise SystemExit

tgz = os.path.join(DEST, "cifar-10-python.tar.gz")
print("downloading", URL, flush=True)
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=300) as r, open(tgz, "wb") as f:
    shutil.copyfileobj(r, f)
print(f"downloaded {os.path.getsize(tgz)} bytes; extracting (safe filter)", flush=True)
with tarfile.open(tgz) as t:
    try:
        t.extractall(DEST, filter="data")        # py3.12+ safe extraction (blocks path traversal)
    except TypeError:
        t.extractall(DEST)                        # older python; trusted canonical source
print("CIFAR-10 ready:", os.path.exists(batch1), "->", batch1, flush=True)
