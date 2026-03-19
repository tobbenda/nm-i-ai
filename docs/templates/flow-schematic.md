# Flow Schematic Template

Reusable data-flow pipeline schematic. Copy and adapt per task.
Track each step's status, code location, and data types between steps.

```
┌═══════════════════════════════════════════════════════════════════════════┐
║                     LOCAL EVALUATOR                                      ║
║                     Code: evaluate.py                ❌ / ✅             ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  1. Load labels → {id: label}                                           ║
║  2. For each sample: run through LOCAL PIPELINE → get prediction        ║
║  3. Collect all predictions                                             ║
║  4. Compute score against true labels                                   ║
║  5. Print score + breakdown                                             ║
║                                                                         ║
╚═════════════════════════════╪═══════════════════════════════════════════╝
                              │ loops over all samples
                              ▼

LOCAL PIPELINE (prep)                        SHARED PIPELINE (API)
═════════════════════                        ═════════════════════

┌─────────────────────────────┐
│ L1. Load raw data from disk │
│                             │
│    Code: ???                │
│    Status: ❌ / ✅          │
└──────────────┬──────────────┘
               │ <raw type>
               ▼
┌─────────────────────────────┐
│ L2. Preprocess              │
│    (scale, normalize, etc)  │
│                             │
│    Code: ???                │
│    Status: ❌ / ✅          │
└──────────────┬──────────────┘
               │ <processed type>
               ▼
┌─────────────────────────────┐
│ L3. Encode for transport    │
│    (base64, JSON, etc)      │
│                             │
│    Code: ???                │
│    Status: ❌ / ✅          │
└──────────────┬──────────────┘
               │ <encoded type>
               ▼
┌─────────────────────────────┐              ┌─────────────────────────────┐
│ L4. POST /predict           │              │ EVAL: server sends same     │
│    {payload}                │              │ request format              │
│                             │              │                             │
│    Code: ???                │              │                             │
│    Status: ❌ / ✅          │              │                             │
└──────────────┬──────────────┘              └──────────────┬──────────────┘
               │ <encoded type>                             │ <encoded type>
               ▼                                            ▼
┌═══════════════════════════════════════════════════════════════════════════┐
║                     API ENDPOINT: POST /predict                          ║
║                     Code: api.py  ❌ / ✅                                ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │ A1. Decode input → internal format                              │    ║
║  │                                                                 │    ║
║  │     Code: ???                                                   │    ║
║  │     Status: ❌ / ✅                                             │    ║
║  └──────────────────────────┬──────────────────────────────────────┘    ║
║                              │ <internal type>                          ║
║                              ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │ A2. Preprocess for model                                        │    ║
║  │     (resize, normalize, etc)                                    │    ║
║  │                                                                 │    ║
║  │     Code: ???                                                   │    ║
║  │     Status: ❌ / ✅                                             │    ║
║  └──────────────────────────┬──────────────────────────────────────┘    ║
║                              │ <model input type>                       ║
║                              ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │ A3. predict(input) → output                                     │    ║
║  │     MODEL / ALGORITHM                                           │    ║
║  │                                                                 │    ║
║  │     Code: model.py                                              │    ║
║  │     Status: ❌ / ⚠️ STUB / ✅                                   │    ║
║  └──────────────────────────┬──────────────────────────────────────┘    ║
║                              │ <prediction type>                        ║
║                              ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │ A4. Return response                                             │    ║
║  │                                                                 │    ║
║  │     Code: api.py                                                │    ║
║  │     Status: ❌ / ✅                                             │    ║
║  └──────────────────────────┬──────────────────────────────────────┘    ║
╚═════════════════════════════╪═══════════════════════════════════════════╝
                              │ JSON response
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SCORE: <metric formula>                                                │
│    LOCAL:  evaluator against local labels     ❌ / ✅                   │
│    EVAL:   server scores against hidden labels                          │
└─────────────────────────────────────────────────────────────────────────┘

CURRENT BASELINE: ???
Status: <what exists, what's missing>
Next: <what to build next>
```

## How to use

1. Copy this file into your task folder (e.g. `docs/M2/`)
2. Replace all `???` with actual code locations and function names
3. Replace `<types>` with actual data types flowing between steps
4. Update ❌ / ✅ as you build each step
5. Fill in the score and status at the bottom
