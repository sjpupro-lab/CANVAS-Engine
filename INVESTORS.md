# CANVAS — Investor Overview

> **The one-line pitch**
> CANVAS is a spatial pattern–based AI engine. It treats language like video, not vectors — giving AI unlimited memory, incremental learning, and a footprint small enough to run on a phone.

![CANVAS](main_hero.png)

---

## 1. The problem

Today's frontier AI is built on a single architectural assumption: **compress language into a fixed-size weight matrix.**

That assumption has three compounding costs:

| Cost | Consequence |
|---|---|
| **Fixed parameters** | Scaling knowledge means retraining. Every update is a full-cost event. |
| **Bounded context** | The window is the memory. Beyond it, the model forgets. |
| **Opaque weights** | You cannot audit, edit, or explain what the model remembers. |

The industry is spending hundreds of billions to paper over these limits — larger models, longer contexts, RAG stacks, vector DBs. None of these change the underlying substrate. They add layers on top of it.

**The substrate itself is the bottleneck.**

---

## 2. The insight

Language has spatial structure. Similar meanings cluster. Different meanings diverge. This structure is usually projected into a vector space and then crushed into weights.

**CANVAS keeps the structure visible.**

Text is rendered onto a 2D grid as a brightness pattern. Each unit of language becomes a frame. Frames stack like video. Memory grows by appending frames — not by retraining a matrix.

The engine then compares patterns directly. No embeddings. No attention. No retraining.

---

## 3. What CANVAS unlocks

| | |
|---|---|
| **Unlimited parameters** | Each new input is a new frame. Frames stack without limit. |
| **Unlimited context** | Bounded by disk, not RAM. |
| **Interpretable memory** | Memory is a visual pattern. You can see what the engine remembers. |
| **Incremental learning** | New knowledge = one frame. No retraining, no fine-tuning. |
| **Edge-native** | Core engine runs on phones, wearables, embedded hardware. |

This is not a better LLM. It is a **different substrate** — one that composes with LLMs rather than replacing them.

---

## 4. vs. Traditional LLMs

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

CANVAS is complementary, not competitive. It is the **memory layer** that LLMs don't have.

---

## 5. Market

CANVAS sits at the intersection of three large and growing markets:

- **AI memory / retrieval infrastructure** — vector DBs, RAG platforms, agent memory
- **On-device AI** — edge inference, mobile AI, embedded assistants
- **Interpretable / auditable AI** — regulated industries, governance, safety

Each of these is tens of billions today and expected to expand as AI moves from chat into persistent agents, devices, and regulated workflows.

**The wedge:** retrieval-heavy, memory-heavy, long-context workloads where pattern persistence matters more than generation. Agent memory, personal AI, on-device assistants, domain knowledge stores.

---

## 6. Use cases

- **Long-context memory for agents** — drop-in memory that doesn't forget, doesn't need a vector DB
- **On-device AI** — incremental updates without cloud retraining
- **Domain knowledge stores** — legal, medical, enterprise corpora updated one frame at a time
- **Interpretable AI research** — a memory substrate you can actually look at
- **Personal AI** — memory that grows with the user and stays on the user's device

---

## 7. Why now

Three converging trends make CANVAS timely:

1. **Context windows are hitting physical and economic walls.** Linear-in-length attention does not scale to "remember everything forever."
2. **On-device AI is becoming a strategic priority** for OS vendors, carriers, and hardware OEMs — and they need lightweight memory primitives.
3. **Interpretability and auditability are moving from research curiosity to regulatory requirement.** Visible memory is a structural answer, not a patch.

CANVAS was designed from the ground up for all three.

---

## 8. Moat

- **Architectural novelty.** The spatial-pattern substrate is a genuinely different approach — not a variation of transformer or SSM architectures.
- **Core IP held privately.** The public repository contains the overview and demos. The core engine is maintained in a private repository.
- **Compounding data advantage.** Because memory is incremental, early deployments produce pattern libraries that improve retrieval quality over time.
- **Footprint advantage.** Running on edge hardware is an architectural property, not an optimization target. It is hard to retrofit into a transformer.

---

## 9. Status

- **Core engine:** working implementation, benchmarked on standard corpora
- **Validated on:** retrieval, recall, and incremental-memory tasks
- **In use internally for:** long-context memory and pattern-matching workloads

> The core engine is maintained in a **private repository**. This public repository contains the overview, concept, and demos.

---

## 10. Roadmap

**Near-term**
- Hardened reference implementation for partner integration
- Reference agents and memory adapters for common LLM stacks
- Edge/mobile SDK (iOS, Android, embedded Linux)

**Mid-term**
- Managed memory service for production agent deployments
- Domain-specific memory packs (legal, medical, enterprise knowledge)
- Developer platform with tooling, observability, and pattern editors

**Long-term**
- CANVAS as the standard memory substrate for on-device and agent AI
- Ecosystem of memory modules, pattern libraries, and third-party integrations

---

## 11. Team

[ To be completed by the founding team. Include: founders, relevant prior work, technical advisors. ]

---

## 12. Use of funds

[ To be completed. Typical allocation for a substrate-infrastructure round:
- Core engineering (engine hardening, SDK, tooling)
- Applied research (benchmarks, new workloads, interpretability)
- Go-to-market (design partners, developer relations)
- Operating runway ]

---

## 13. The ask

We are raising a **[ round size / round stage ]** round to harden the engine, ship the SDK, and land the first production design partners.

If you believe:
- That memory, not parameters, is the real scaling axis
- That on-device AI is where the next decade of usage will live
- That interpretable AI is a requirement, not a nice-to-have

— we would like to talk.

---

## Contact

For collaboration, licensing, or investment inquiries, please open an issue or reach out via the repository owner's profile.

---

## Appendix — Further reading

- [README.md](README.md) — technical and conceptual overview
- [PITCH_DECK.md](PITCH_DECK.md) — slide deck (Marp-compatible)
- [LICENSING.md](LICENSING.md) — licensing strategy (open-core structure, IP, moat)
- [README_KO.md](README_KO.md) — Korean overview
- [INVESTORS_KO.md](INVESTORS_KO.md) — Korean investor overview
