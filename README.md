# CANVAS

### A spatial pattern–based AI engine.

![CANVAS](main_hero.png)

> Language is treated like video, not vectors.
> Memory is a picture, not a weight matrix.

---

## The idea

Traditional LLMs compress language into a fixed-size weight matrix.
To add knowledge, you retrain. To inspect memory, you can't — it's opaque.

**CANVAS does the opposite.**

Text is rendered onto a spatial grid as a brightness pattern.
Each unit of language becomes a frame. Frames stack like video.
Memory grows by appending frames, not by retraining a matrix.

---

## What it enables

| | |
|---|---|
| **Unlimited parameters** | Each new input is a new frame. Frames stack without limit. |
| **Unlimited context** | Bounded by disk, not RAM. |
| **Interpretable** | The engine's memory is a visual pattern. You can see what it remembers. |
| **Incremental** | New knowledge = one frame. No retraining. |
| **Lightweight** | Core engine runs on phones, edge devices, embedded hardware. |

---

## Visual overview

![Spatial encoding](visualization_1.png)

Language is encoded as a pattern on a 2D grid.
Similar inputs produce similar patterns. Different inputs produce different patterns.
The engine compares patterns directly — no embeddings, no attention.

![Matching pipeline](visualization_2.png)

Retrieval is a two-stage process: a fast coarse filter, then a precise match.
Both stages operate on the visual pattern itself.

---

## Why this matters

The core bet: **language has enough spatial structure that you can treat it like a signal, not a symbol.**

If the bet holds, you get an AI substrate that:

- Grows with usage instead of being retrained
- Remembers forever instead of forgetting context
- Shows you its memory instead of hiding behind weights
- Fits on a phone instead of a data center

---

## vs. Traditional LLMs

```
                   Traditional LLM              CANVAS
  ─────────────────────────────────────────────────────────────
  Representation   Token → vector → matrix     Pattern on a grid
  Parameters       Fixed-size matrix           Unlimited frame stack
  Context          Bounded window              Disk-bound (effectively unlimited)
  Interpretability Opaque weights              Visible pattern
  Learning         Full retrain / fine-tune    Incremental: one frame
  Footprint        Data-center scale           Embedded-friendly
```

CANVAS is not a drop-in replacement for a generative LLM.
It is a **substrate** — best suited to retrieval-heavy, memory-heavy, and long-context tasks where pattern persistence matters more than generation.

---

## Status

- Core engine: working implementation, benchmarked on standard corpora
- Verified on retrieval, recall, and incremental-memory tasks
- Used internally for long-context memory and pattern-matching workloads

The **core engine is maintained in a private repository.**
This repository contains the public overview, concept, and demos.

---

## Use cases

- Long-context memory for agents
- Retrieval substrates that grow with usage
- On-device AI with incremental updates
- Research on interpretable / visual AI
- Domain-specific knowledge stores that don't need retraining

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

- **GitHub:** https://github.com/sjpupro-lab/CANVAS-Engine
- **Team:** SJPU TEAM
- **Email:** sjpupro@gmail.com
- **Inquiries:** https://github.com/sjpupro-lab/CANVAS-Engine/issues

For collaboration, licensing, or investment inquiries, please reach out via the channels above.
