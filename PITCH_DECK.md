---
marp: true
theme: default
paginate: true
backgroundColor: "#0b0b0f"
color: "#eaeaea"
style: |
  section { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  h1 { color: #ffffff; letter-spacing: -0.02em; }
  h2 { color: #ffffff; border-bottom: 1px solid #333; padding-bottom: 0.3em; }
  code, pre { background: #141419; color: #eaeaea; }
  strong { color: #ffffff; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #333; padding: 0.5em 1em; }
---

<!-- _class: lead -->

# **CANVAS**

### A spatial pattern–based AI engine

Language is treated like video, not vectors.
Memory is a picture, not a weight matrix.

---

## The problem

Today's AI is built on one architectural bet:
**compress language into a fixed-size weight matrix.**

That bet has three compounding costs:

- **Fixed parameters** → scaling = retraining
- **Bounded context** → the window *is* the memory
- **Opaque weights** → you cannot audit, edit, or explain

The industry is spending billions to paper over the substrate.
**The substrate itself is the bottleneck.**

---

## The insight

Language has spatial structure.
Usually that structure is projected into vectors and crushed into weights.

### CANVAS keeps the structure **visible.**

- Text → brightness pattern on a 2D grid
- Each unit of language → a frame
- Frames stack like video
- Memory grows by **appending frames**, not by retraining a matrix

---

## How it works

![bg right:45% 90%](visualization_1.png)

Language is encoded as a pattern on a 2D grid.

- Similar inputs → similar patterns
- Different inputs → different patterns
- The engine compares patterns **directly**

No embeddings. No attention. No retraining.

---

## Matching pipeline

![bg right:45% 90%](visualization_2.png)

Retrieval is two stages:

1. **Fast coarse filter** on the pattern
2. **Precise match** on the pattern

Both stages operate on the visual representation itself — not on weights, not on vectors.

---

## What this unlocks

| | |
|---|---|
| **Unlimited parameters** | Each input is a new frame. No cap. |
| **Unlimited context** | Bounded by disk, not RAM. |
| **Interpretable** | Memory is a visual pattern. You can *see* it. |
| **Incremental** | New knowledge = one frame. No retraining. |
| **Lightweight** | Runs on phones, wearables, embedded. |

---

## vs. Traditional LLMs

```
                 Traditional LLM          CANVAS
  ─────────────────────────────────────────────────────
  Representation Token → vector → matrix  Grid pattern
  Parameters     Fixed-size matrix        Unlimited stack
  Context        Bounded window           Disk-bound
  Interpret.     Opaque weights           Visible pattern
  Learning       Retrain / fine-tune      One frame
  Footprint      Data-center              Embedded
```

> CANVAS is **complementary**, not competitive.
> It is the memory layer LLMs don't have.

---

## Why now

1. **Context windows are hitting walls** — linear attention can't "remember forever."
2. **On-device AI is a strategic priority** — OS vendors, carriers, OEMs all need lightweight memory primitives.
3. **Interpretability is moving from research to regulation** — visible memory is a structural answer, not a patch.

CANVAS was designed for all three, from day one.

---

## Market

Three large, growing markets — one substrate.

- **AI memory / retrieval infra** — vector DBs, RAG, agent memory
- **On-device AI** — edge, mobile, embedded
- **Interpretable / auditable AI** — regulated industries, governance

**Wedge:** retrieval-heavy, memory-heavy, long-context workloads
where pattern persistence > generation.

---

## Use cases

- **Agent memory** — drop-in, doesn't forget, no vector DB needed
- **On-device AI** — incremental updates without cloud retraining
- **Domain knowledge stores** — legal, medical, enterprise — one frame at a time
- **Personal AI** — memory that grows with the user, stays on-device
- **Interpretable AI research** — a memory substrate you can look at

---

## Moat

- **Architectural novelty** — not a transformer or SSM variant
- **Core IP held privately** — public repo = overview; core = private
- **Compounding data advantage** — incremental memory builds pattern libraries over time
- **Footprint advantage** — edge execution is an architectural property, not an optimization

---

## Status

- **Core engine:** working implementation, benchmarked on standard corpora
- **Validated:** retrieval, recall, incremental-memory tasks
- **In internal use:** long-context memory & pattern-matching workloads

Core engine maintained in a **private repository.**
Public repo = overview, concept, demos.

---

## Roadmap

**Near-term**
Hardened reference impl · LLM-stack adapters · edge/mobile SDK

**Mid-term**
Managed memory service · domain memory packs · developer platform

**Long-term**
Standard memory substrate for on-device and agent AI
Ecosystem of modules, pattern libraries, integrations

---

## Team

<!-- To be completed by the founding team.
     Include: founders, relevant prior work, technical advisors. -->

[ Founders · Prior work · Advisors ]

---

## The ask

Raising **[ round size / stage ]** to:

- Harden the engine
- Ship the SDK
- Land the first production design partners

> If you believe memory — not parameters — is the real scaling axis,
> we would like to talk.

---

<!-- _class: lead -->

# Thank you

**CANVAS** — A spatial pattern–based AI engine

Collaboration · Licensing · Investment

- **GitHub:** https://github.com/sjpupro-lab/CANVAS-Engine
- **Team:** SJPU TEAM
- **Email:** sjpupro@gmail.com
- **Inquiries:** https://github.com/sjpupro-lab/CANVAS-Engine/issues
