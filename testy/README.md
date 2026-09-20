# testy — porównanie jakości: Marcin116 (INT4-embed) vs born2bewild (INT8-embed)

End-to-end test A/B dla dwóch kwantyzacji `Qwen3.8-27B-W4A16-AutoRound-fast`,
różniących się **tylko** tabelą embeddingów (`embed_tokens` INT8 → INT4).

## Metodyka (reproducible)

- **Metryka:** perpleksja (PPL) — suma NLL z `prompt_logprobs` z vLLM
  (faktyczny token promptu, nie top-1; 0 braków logprobów).
- **Korpus:** `corpus.txt` — 8 artykułów Wikipedii (PL-free, en-wiki),
  ~102 000 tokenów. **Ten sam plik** dla obu modeli — czysty A/B.
- **Tokenizacja:** `tokenizer.json` z repo (r1 i r2 mają identyczny, ten sam hash).
- **Parametry:** `temperature=0`, `max_tokens=1`, okna 2048 tokenów,
  concurrency 1 (serwer służy normalnie w tym czasie).
- **Uruchomienie:**
  ```bash
  python testy/ppl.py            # pełny korpus
  python testy/ppl.py smoke      # 1 okno, walidacja
  ```
  Skrypt auto-detekuje model z `/v1/models` (id + `root`) **oraz wersję
  vLLM z `/version`** (trafia do wyników i nazwy pliku) i zapisuje wynik
  do `results/r<repo>_<vllm>.json`.
- **Kontrola:** `quality.py` — tensorowy test kwantyzacji embeddingów
  (dekwantyzacja INT4/INT8 vs BF16 bazowy; 256 wierszy, 1,31M elementów).

## Wyniki (stan: 2026-09-20)

| model (served `root`) | embedding | vLLM | PPL (korpus ~102k tok.) | wynik |
|---|---|---|---|---|
| `born2bewild/...-fast` (r2) | INT8 | stary build | 4.7138 | `results/r2_born2bewild_INT8embed_pre_swap.json` |
| `born2bewild/...-fast` (r2) | INT8 | **`0.29.1rc1.dev434+g27757dde0`** | **4.7136** | `results/r2_born2bewild_INT8embed_vllm0.29.1rc1dev434.json` |
| `Marcin116/...-fast-embed-int4` (r1) | INT4 | `0.29.1rc1.dev434` | *oczekiwany* | `results/r1_marcin116_INT4embed_vllm0.29.1rc1dev434.json` ⏳ |

**Pomiar na vLLM 0.29.1rc1.dev434 (2026-09-20): 510 tok/s, ~201 s, 0 retry, 0 braków.**

> **Kontrola neutralności vLLM:** r2 mierzony na starym i nowym buildzie
> vLLM daje PPL 4.7138 / 4.7136 (Δ 0.0002, w granicy szumu pomiaru) —
> podmiana vLLM nie wpływa na wynik. (Wersja buildu sprzed podmiany nie
> została wtedy zapisana — harness nagrał ją od 2026-09-20; r1 będzie
> mierzony na tym samym, nowym buildzie, więc porównanie r1 vs r2 jest
> porównaniem na identycznym runtime.)

### Interpretacja (kryteria)

| PPL r1 vs r2 | ocena |
|---|---|
| Δ < 1% (PPL ≤ ~4.75) | różnica niewymierzalna — INT4-embed da się nie zauważyć |
| Δ 1–7% (~4.75–5.1) | realna, subtelna — w typowym użyciu raczej niewidoczna |
| Δ > 15% (PPL ≥ ~5.5) | odczuwalna degradacja |

Kontekst: reszta modelu (95,3% parametrów, W4) jest **identyczna bit-po-bicie**;
`lm_head` jest osobny (`tie_word_embeddings=False`), więc wpływ INT4-embed
ogranicza się do reprezentacji wejścia. Tensorowo: r1 vs r2 cosine 0.986 /
per-token median 0.9915 (kierunki wektorów dobrze zachowane).

## Kontekst pochodzenia (potwierdzone)

- r1 to minimalny derivat r2: różnice tylko w `README.md`, `config.json`
  (`num_bits`: 4 vs 8) i shard-6 (embedding).
- Proweniencja: r1 lepiej tłumaczy się jako **re-quantyzacja z BF16 bazy**
  (92,5% rozmytych zaokrągleń zgadza się z zaokrągleniem BF16 vs 7,5% z INT8
  r2), nie jako INT8→INT4 z r2 — skala grupy r1 ≈ 18,1× skali r2.
