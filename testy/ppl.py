#!/usr/bin/env python
# Perplexity harness for an OpenAI-compatible vLLM endpoint.
# Usage: ppl.py [smoke]   -- "smoke" runs a single 2048-token window for validation.
import json, math, sys, time
import requests
from tokenizers import Tokenizer

BASE = "http://192.168.1.124:8002/v1"
TOK = Tokenizer.from_file("r2_tokenizer.json")
MODEL = requests.get(BASE + "/models", timeout=30).json()["data"][0]
MODEL_ID = MODEL["id"]
ROOT = MODEL.get("root")
try:
    VLLM_VERSION = requests.get("http://192.168.1.124:8002/version", timeout=30).json().get("version")
except Exception:
    VLLM_VERSION = None
WIN = 2048

text = open("corpus.txt").read()
ids = TOK.encode(text).ids
print("model_id=%s  served_root=%s  vllm=%s" % (MODEL_ID, ROOT, VLLM_VERSION))
print("corpus: %d chars -> %d tokens; windows of %d -> %d windows" %
      (len(text), len(ids), WIN, (len(ids) + WIN - 1) // WIN))

def chunk_ppl(chunk, k=1):
    last_err = None
    for attempt in range(5):
        try:
            r = requests.post(BASE + "/completions",
                              json={"model": MODEL_ID, "prompt": chunk,
                                    "max_tokens": 1, "temperature": 0,
                                    "prompt_logprobs": k},
                              timeout=600)
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
            print("    retry %d after %r" % (attempt + 1, e), flush=True)
    else:
        raise last_err
    plp = r.json()["choices"][0]["prompt_logprobs"]
    tot, n, miss = 0.0, 0, 0
    for i in range(1, len(chunk)):
        e = plp[i]
        if e is None:
            continue
        key = str(chunk[i])
        if key in e:
            tot += e[key]["logprob"]
            n += 1
        else:
            miss += 1
    return tot, n, miss

t0 = time.time()
if "smoke" in sys.argv:
    tot, n, miss = chunk_ppl(ids[:WIN])
    print("SMOKE: n=%d miss=%d avg_logprob=%.4f ppl=%.3f (%.1fs)" %
          (n, miss, tot / n, math.exp(-tot / n), time.time() - t0))
    # show a few positions for sanity
    r = requests.post(BASE + "/completions",
                      json={"model": MODEL_ID, "prompt": ids[:8],
                            "max_tokens": 1, "temperature": 0,
                            "prompt_logprobs": 1},
                      timeout=60)
    for i, e in enumerate(r.json()["choices"][0]["prompt_logprobs"]):
        if e is None:
            continue
        key = str(ids[i])
        print("  pos %d id=%d in_dict=%s lp=%s" % (i, ids[i], key in e, e.get(key, {}).get("logprob")))
    sys.exit(0)

tot, n, miss, t = 0.0, 0, 0, 0.0
for w in range(0, len(ids), WIN):
    chunk = ids[w:w + WIN]
    st = time.time()
    tot_w, n_w, miss_w = chunk_ppl(chunk)
    if miss_w:
        tot_w, n_w, miss_w = chunk_ppl(chunk, k=50)  # fallback: wider top-k
    tot += tot_w
    n += n_w
    miss += miss_w
    if (w // WIN) % 10 == 0:
        el = time.time() - t0
        print("  window %4d/%4d  ppl_so_far=%.3f  (%.1fs, %.0f tok/s)" %
              (w // WIN, len(ids) // WIN, math.exp(-tot / max(n, 1)),
               el, n / max(el, 1e-9)), flush=True)

ppl = math.exp(-tot / n)
out = {"served_model_id": MODEL_ID, "served_root": ROOT,
       "vllm_version": VLLM_VERSION,
       "corpus_chars": len(text), "tokens": len(ids),
       "tokens_scored": n, "misses": miss,
       "total_nll": -tot, "perplexity": ppl, "wall_seconds": round(time.time() - t0, 1)}
print("\n" + json.dumps(out, indent=1))
ver = str(VLLM_VERSION or "unknown").replace("+", "_").replace(".", "_")
open("ppl_result_%s_vllm%s.json" % ((ROOT or MODEL_ID).replace("/", "_"), ver), "w").write(json.dumps(out, indent=1))
