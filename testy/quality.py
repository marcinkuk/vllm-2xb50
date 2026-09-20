import json, urllib.request, numpy as np, struct

def rng(url, start, length):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, start + length - 1)})
    r = urllib.request.urlopen(req)
    d = r.read()
    assert r.status == 206 and len(d) == length, (r.status, len(d))
    return d

def bf16(d):
    u16 = np.frombuffer(d, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def hdr(url):
    d8 = rng(url, 0, 8)
    L = struct.unpack("<Q", d8[:8])[0]
    d = rng(url, 8, L)
    j = json.loads(d.decode())
    return L, j

R, C = 256, 5120

# ---- base: Qwen/Qwen3.8-27B, shard 3 ----
UB = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/model-00003-of-00018.safetensors"
Lb, jb = hdr(UB)
ob = jb["model.language_model.embed_tokens.weight"]["data_offsets"]
base_raw = rng(UB, 8 + Lb + ob[0], R * C * 2)
base = bf16(base_raw).astype(np.float64).reshape(R, C)
print("base head check: first row first 8 =", base[0, :8].round(4).tolist())

u1 = "https://huggingface.co/Marcin116/Qwen3.8-27B-W4A16-AutoRound-fast-embed-int4/resolve/main/model-00006-of-00007.safetensors"
u2 = "https://huggingface.co/born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast/resolve/main/model-00006-of-00007.safetensors"
m1 = json.load(open("r1_shard6_header.json")); h1 = m1["tensors"]; D1 = 8 + m1["header_bytes"]
m2 = json.load(open("r2_shard6_header.json")); h2 = m2["tensors"]; D2 = 8 + m2["header_bytes"]

s1 = bf16(rng(u1, D1 + h1["model.language_model.embed_tokens.weight_scale"]["data_offsets"][0], R * 40 * 2)).astype(np.float64).reshape(R, 40)
s2 = bf16(rng(u2, D2 + h2["model.language_model.embed_tokens.weight_scale"]["data_offsets"][0], R * 40 * 2)).astype(np.float64).reshape(R, 40)
pw1 = rng(u1, D1 + h1["model.language_model.embed_tokens.weight_packed"]["data_offsets"][0], R * 640 * 4)
pw2 = rng(u2, D2 + h2["model.language_model.embed_tokens.weight_packed"]["data_offsets"][0], R * 1280 * 4)
w1 = np.frombuffer(pw1, dtype=np.uint32).reshape(R, 640)
w2 = np.frombuffer(pw2, dtype=np.uint32).reshape(R, 1280)

a = np.zeros((R, C), np.int64)
for r in range(R):
    for c in range(C):
        a[r, c] = ((int(w1[r, c // 8]) >> (4 * (c % 8))) & 0xF) - 8
a_dec = a * np.repeat(s1, 128, axis=1)

b = np.zeros((R, C), np.int64)
for r in range(R):
    for c in range(C):
        b[r, c] = ((int(w2[r, c // 4]) >> (8 * (c % 4))) & 0xFF) - 128
b_dec = b * np.repeat(s2, 128, axis=1)

def cos(x, y):
    return float(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y)))

def relRMS(x, y):
    return float(np.sqrt(((x.ravel() - y.ravel()) ** 2).mean()) / np.sqrt((y.ravel() * y.ravel()).mean()))

print()
print("=== %d embedding rows (1.31M elements) vs BF16 ground truth ===" % R)
print("  r2 INT8 vs base: cosine=%.5f  relRMS=%6.3f%%" % (cos(b_dec, base), relRMS(b_dec, base) * 100))
print("  r1 INT4 vs base: cosine=%.5f  relRMS=%6.3f%%" % (cos(a_dec, base), relRMS(a_dec, base) * 100))

print()
print("=== r1 vs r2 directly (the actual model-level difference) ===")
print("  cosine(r1,r2)=%.5f  relRMS(r1,r2)=%6.3f%%" % (cos(a_dec, b_dec), relRMS(a_dec, b_dec) * 100))

print()
print("=== per-token (row) quality: relative error of each token vector ===")
def rowerr(d):
    num = np.sqrt(((d - base) ** 2).sum(1))
    den = np.sqrt((base ** 2).sum(1))
    return num / den
def rowcos(d):
    dd = d * base
    num = dd.sum(1)
    den = np.sqrt((d ** 2).sum(1)) * np.sqrt((base ** 2).sum(1))
    return num / den
for name, d in (("r2 INT8", b_dec), ("r1 INT4", a_dec)):
    e = rowerr(d)
    c = rowcos(d)
    print("  %-8s per-token rel err: median=%6.3f%% p99=%6.3f%% max=%6.3f%% | per-token cosine: median=%.5f p1=%.5f min=%.5f" % (
        name, np.median(e) * 100, np.percentile(e, 99) * 100, e.max() * 100, np.median(c), np.percentile(c, 1), c.min()))

print()
print("=== error budget within the model ===")
err1b = np.linalg.norm(a_dec - base)
err2b = np.linalg.norm(b_dec - base)
print("  |r1-base| / |r2-base| = %.2fx  (r1 embedding carries %d times more noise than r2's)" % (err1b / err2b, err1b / err2b))
# embedding size vs whole model
n_emb = 248320 * 5120
n_tot = 27.2e9
print("  embedding = %.1f%% of total params; the other %.1f%% is W4 in BOTH repos" % (100 * n_emb / n_tot, 100 * (1 - n_emb / n_tot)))
