# CANVAS — Licensing Strategy

> This document describes the **licensing strategy** for the CANVAS engine and related materials.
> It is a strategy document, not a legal contract. For actual license terms, see the individual license files and commercial agreements.

---

## 1. Goals

The licensing strategy is designed to satisfy four simultaneous goals:

1. **Protect the core.** The engine is the crown jewel. It should never become freely redistributable in a form that undercuts commercial use.
2. **Seed the ecosystem.** Developers, researchers, and students must be able to learn, prototype, and publish on top of CANVAS without friction.
3. **Capture value from production use.** Commercial deployments — especially at scale, on-device, or behind paywalls — must flow back as revenue.
4. **Keep the investor story clean.** Clear IP ownership, clear revenue path, no accidental copyleft contamination.

---

## 2. Tiered license structure

CANVAS uses a **three-tier open-core structure**:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Tier 3 — Commercial                                             │
  │  ───────────────────                                             │
  │  Production SDKs · on-device · embedded · managed service        │
  │  License: CANVAS Commercial License (paid)                       │
  │  IP:      Proprietary, held privately                            │
  ├──────────────────────────────────────────────────────────────────┤
  │  Tier 2 — Source-Available (Community)                           │
  │  ─────────────────────────────────────                           │
  │  Reference engine for research, evaluation, non-production use   │
  │  License: CANVAS Community License (BSL / PolyForm-Noncommercial)│
  │  IP:      Proprietary, source visible                            │
  ├──────────────────────────────────────────────────────────────────┤
  │  Tier 1 — Public                                                 │
  │  ────────────                                                    │
  │  Overview, concept, demos, visualizations, adapters              │
  │  License: MIT                                                    │
  │  IP:      Openly shared                                          │
  └──────────────────────────────────────────────────────────────────┘
```

### Tier 1 — Public (MIT)

- **What it covers:** this repository — README, pitch deck, investor materials, conceptual visualizations, and any thin reference adapters/demos we publish.
- **Why MIT:** maximum distribution, zero friction for evaluators, minimum legal overhead. No copyleft risk for the community.
- **What it does *not* cover:** the core engine implementation, the pattern-matching internals, or any optimized production code.

### Tier 2 — Source-Available (Community)

- **Intended license family:** **Business Source License (BSL 1.1)** with a non-commercial restriction, or alternatively **PolyForm Noncommercial 1.0.0**.
- **What it covers:** a reference implementation of the engine — enough for researchers, developers, and design partners to **read, run, benchmark, and build on** — but not enough to ship commercially.
- **Key clauses:**
  - Permitted: research, personal use, internal evaluation, academic publication, non-commercial open-source projects
  - Not permitted: production use, SaaS hosting, on-device shipping, redistribution as a competing product
  - **Automatic relicensing:** each BSL-released version transitions to Apache 2.0 after **N years** (default: 4). This gives the community long-term confidence without undermining near-term commercial value.
- **Why BSL (not AGPL/GPL):**
  - AGPL scares enterprise customers and blocks common commercial patterns (embedding, SaaS)
  - BSL is explicit: non-commercial use is welcome, commercial use requires a paid license
  - Precedent: HashiCorp, MongoDB, CockroachDB, Sentry, Redis Labs

### Tier 3 — Commercial (Proprietary)

- **What it covers:**
  - Production-hardened engine builds
  - Optimized on-device / embedded SDKs (iOS, Android, embedded Linux, RTOS)
  - Managed memory service / hosted CANVAS
  - Domain-specific memory packs
  - Enterprise support, SLAs, and indemnification
- **License:** **CANVAS Commercial License** — negotiated, per-deployment or per-seat.
- **Packaging:** see §5.

---

## 3. IP ownership and repository policy

- **Public repo (`canvas-engine`)** — MIT, can be freely forked.
- **Core engine repo** — **private**, proprietary. Never pushed to public remotes. Access limited to core team and contractually-bound partners.
- **All contributions from non-employees** must be covered by a **Contributor License Agreement (CLA)** granting the company the right to relicense contributions commercially. Without a CLA, contributions are not accepted into tiers 2 or 3.
- **No copyleft dependencies** (GPL, AGPL, LGPL-with-static-linking) are permitted in the core engine. This is enforced as a build-time check.

---

## 4. Trademark

- **"CANVAS"** and the CANVAS logo are trademarks of the company.
- MIT-licensed code in the public repo does **not** grant trademark rights. Downstream users may fork the code under MIT but may not ship a product called "CANVAS" or any confusingly similar mark without a trademark license.
- This is the standard OSS-pattern trademark carve-out and is explicitly compatible with MIT (which covers copyright, not trademark).

---

## 5. Commercial packaging

Commercial tiers are priced on the axes that align with customer value and our marginal cost:

| Axis | Typical pricing signal |
|---|---|
| **Deployment surface** | Cloud · on-prem · on-device · embedded |
| **Scale** | Frames stored · QPS · device count |
| **Integration depth** | SDK only · managed service · white-label |
| **Support** | Community · standard · enterprise (24/7, SLA) |

Indicative package shapes (to be finalized):

- **Startup** — flat per-year license for early-stage companies below a revenue/funding threshold
- **Production** — usage-based (frames / QPS / devices) with volume tiers
- **Enterprise** — negotiated, includes on-prem, support, SLA, and indemnification
- **OEM / embedded** — per-unit royalty for hardware shipping with CANVAS on-device
- **Academic / research** — free under Tier 2 with publication requirements

---

## 6. Enforcement & compliance

- **License verification** is built into the commercial SDK (signed license keys, periodic validation for online deployments, offline activation for air-gapped).
- **Telemetry is opt-in** and never collects user content. The license-check channel carries only license metadata.
- **Audit rights** are included in enterprise contracts, exercised only on reasonable suspicion of breach.
- **Open-source compliance** on our side: every dependency in the core engine is vetted for license compatibility before merge.

---

## 7. How this ties to the investor story

- **Clean IP.** Single company owns the core. CLAs ensure no external contributor can assert claims. No copyleft contamination.
- **Defensible moat.** BSL prevents cloud hyperscalers from offering a competing managed service on top of our own source.
- **Dual revenue motion.** Bottom-up adoption via Tiers 1–2 feeds top-down commercial contracts in Tier 3.
- **Optional future open-sourcing.** The BSL sunset clause gives a credible "eventually open" story without sacrificing near-term revenue — useful for regulatory, public-sector, and community narratives.
- **Exit compatibility.** The structure is familiar to acquirers (mirrors HashiCorp, MongoDB, Confluent) and does not require unwinding at acquisition.

---

## 8. Current status

- **Tier 1 (Public / MIT):** live — see [LICENSE](LICENSE).
- **Tier 2 (Source-Available):** license text to be finalized before first external release of the reference engine.
- **Tier 3 (Commercial):** template agreement to be finalized with counsel before first paid design partner.

---

## 9. Open questions for counsel

The following items require review by qualified IP and commercial counsel in the company's primary jurisdiction(s) before execution:

- Final choice between **BSL 1.1** and **PolyForm Noncommercial** for Tier 2
- BSL **change date** (default: +4 years) and **change license** (default: Apache 2.0)
- **CLA** form — individual and corporate
- **Trademark** registration strategy (jurisdictions, classes)
- **Export control** classification for on-device / embedded builds
- **Patent strategy** for the core spatial-pattern method (defensive filings, prior-art strategy)

---

## 10. References

- Business Source License: https://mariadb.com/bsl11/
- PolyForm licenses: https://polyformproject.org/
- HashiCorp BSL precedent: https://www.hashicorp.com/blog/hashicorp-adopts-business-source-license
- MongoDB SSPL discussion (for contrast): https://www.mongodb.com/licensing/server-side-public-license

---

## Related documents

- [LICENSE](LICENSE) — the MIT license currently covering the public repo (Tier 1)
- [README.md](README.md) — project overview
- [INVESTORS.md](INVESTORS.md) — investor overview
- [PITCH_DECK.md](PITCH_DECK.md) — slide deck
- [LICENSING_KO.md](LICENSING_KO.md) — Korean version of this document

---

## Contact

For commercial licensing (Tier 3), CLA intake, trademark requests, or any licensing-related inquiry:

- **GitHub:** https://github.com/sjpupro-lab/CANVAS-Engine
- **Team:** SJPU TEAM
- **Email:** sjpupro@gmail.com
- **Inquiries:** https://github.com/sjpupro-lab/CANVAS-Engine/issues
