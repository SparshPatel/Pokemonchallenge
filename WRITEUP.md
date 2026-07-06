# PTCG AI Battle — Model Summary / Write-up

> **Competition:** The Pokémon Company – PTCG AI Battle Challenge (Simulation)
> **Fill before submitting:** items marked `<FILL>` (name, location, email, team name, leaderboard rank).

---

## A1. Background on you / your team

- **Competition Name:** The Pokémon Company - PTCG AI Battle Challenge Simulation
- **Team Name:** `<FILL>`
- **Leaderboard Score:** `<FILL — Simulation skill rating at final deadline>`
- **Leaderboard Place:** `<FILL>`

Team members:
- **Name:** `<FILL>` — **Location:** `<FILL>` — **Email:** `<FILL>`

## A2. Background

- **Academic / professional background:** quantitative research / machine learning and optimization.
- **Prior experience that helped:** search-based decision making under uncertainty (stochastic
  control, Monte-Carlo / beam search), and disciplined out-of-sample evaluation to avoid
  overfitting to noisy signals.
- **Why this competition:** it is a clean sequential-decision problem — a deterministic-rules game
  with hidden information and shuffle variance — that rewards good planning and robust evaluation
  rather than large-scale model training.

## A3. Summary

Our agent is a **search-based planner**, not a trained neural network. Each decision point is
solved by a **depth-limited beam search over determinized game states** (hidden information is
sampled), scored by a **hand-tuned linear evaluation function** over game features (prize lead,
damage dealt/taken, board readiness, energy, threat). A **rule-based policy** provides a fast,
always-legal fallback so the agent never forfeits on an exception or a time-out. The deck is a
mono-**Fighting** aggro shell built around **Koraidon ex** and **Mega Lucario ex** with a heavy
consistency package (Ultra Ball, Buddy-Buddy Poffin, Boss's Orders, draw supporters). The agent is
**CPU-only** and spends roughly **0.12 s of think time per decision**; there is no offline training
phase — "tuning" was done by measured self-play. Against a diverse field of opponent decks and
opponent policies it wins **≈ 0.83** of games.

## A4. Features / Engineering

The evaluation function scores an end-of-turn state as a weighted sum of interpretable features
(higher = better for us). Approximate weights used in the shipped version:

| Feature | Weight | Meaning |
|---|---:|---|
| `win` | 100000 | terminal win (dominant) |
| `no_active` | 220 | penalise having no legal Active Pokémon (near-loss) |
| `prize` | 120 | prize-card differential (race position) |
| `opp_dmg` | 70 | damage we have placed on the opponent's Active |
| `opp_threat` | 60 | opponent's ability to KO us next turn (defensive) |
| `setup_ko` | 45 | we are set up to KO next turn |
| `opp_bench_dmg` | 35 | pressure on the opponent's bench |
| `my_hp` | 22 | our Active's remaining HP |
| `bench_setup_ko` | 22 | benched attacker ready to take over |
| `my_ready` | 18 | our Active is energised / can attack |
| `bench` | 12 | healthy bench width (resilience) |
| `energy` | 6 | attached-energy progress |
| `hand` | 2 | hand size (tie-breaker) |

**Deck as "features".** The single most important design lever was the **deck list**, because it
determines *which* matchups we win. The final list keeps **18 Fighting energy** (load-bearing —
reducing it collapses the win rate) plus a dense draw/search package so the aggressive line comes
online consistently:
- **Attackers:** Koraidon ex ×3, Mega Lucario ex ×2 (+ Riolu ×2), Bloodmoon Ursaluna ×3, Regirock
  ex ×2, Ogerpon ex ×2, Fezandipiti ex ×2, Ting-Lu ×2.
- **Consistency:** Ultra Ball ×4, Buddy-Buddy Poffin ×4, Pokégear ×2, Boss's Orders ×3,
  Lillie's Determination ×2, Switch ×2, Air Balloon ×2, Night Stretcher ×1, Prime Catcher ×1
  (ACE SPEC), Tarragon ×1, Lively Stadium ×1, Lucky Helmet ×1.

**Feature selection method.** Weights and deck slots were chosen by **paired self-play measurement**
(head-to-head, same deck, low-variance paired comparison) and confirmed on a **1000-deck field
evaluation** against five opponent policies (random, greedy, tempo, strong, pivot). We only promoted
a change if it improved the *field* score, not merely a head-to-head mirror.

## A5. Agent / "Training" Methods

There is **no gradient-based training**. The agent is a classical game-playing pipeline:

1. **Determinized beam search (`PlannerV2`).** Parameters: `beam_width = 4`, `max_depth = 6`,
   `n_determinizations (k) = 2`, `max_nodes = 300`, `max_think_s = 0.12`. At each of our decision
   points we expand candidate action sequences, sample the hidden state `k` times to handle
   imperfect information, roll out to the end of our turn, and score leaves with the evaluation
   function above. The best first action is played.
2. **Rule-based fallback (`rules.py`).** A fast heuristic policy that is guaranteed legal. It is used
   when the planner is disabled, exceeds its node/time budget, or would return an invalid choice.
3. **Safety wrapper (`main.py`).** Every code path is wrapped so the agent **never raises**; on any
   error it returns the lowest `minCount` option indices. In this competition an exception is scored
   as a forfeit, so "never crash" is a first-class requirement.

Ensembling: none. The planner + rules fallback act as a single policy (planner primary, rules
safety net).

## A6. Interesting findings

- **The 0.83 ceiling is structural.** Independent efforts — deck-consistency tuning, evaluation-
  weight tuning, and full archetype swaps — all converged on ≈ 0.829–0.833 field win-rate. The
  residual losses are driven by shuffle variance (bad opening hands, energy droughts, prize-race
  luck), not by strategic mistakes the current search can fix.
- **Head-to-head mirrors are a misleading signal.** A variant that beat the baseline 0.650 in a
  planner-vs-planner mirror still *tied* it (0.829) on the real diverse field. Lesson: **always
  gate promotions on field evaluation**, never on mirror head-to-head.
- **ACE SPEC deck-legality trap.** Master Ball, Precious Trolley and Prime Catcher are all ACE SPEC
  — a deck may contain **at most one ACE SPEC card total**. Over-including them makes the engine
  **silently reject** the deck (start returns a null observation → every game is an instant "draw"
  in 0 seconds), which is easy to misread as a code bug.
- **Archetype exploration confirmed Fighting is best for this field.** We built and measured three
  alternative archetypes: Mega Kangaskhan ex (Colorless, any-energy + draw) → **0.18** H2H;
  Gouging Fire ex (Fire OHKO) → **0.42** H2H; Zacian ex (Metal, self-accelerating) → best of the
  three at 0.55 H2H but only **0.774** on the field vs the Fighting shell's **0.833**. The Fighting
  aggro shell out-tempos every alternative against the mono-type Basic-attacker field.
- **Load-bearing cards.** Cutting the 18th–15th Fighting energy, or removing Night Stretcher,
  produced large negative swings — consistency of the aggressive line matters more than adding
  "tech" cards.

## A7. Simple features and methods

- A **rules-only** agent (no search) plus the same Fighting deck already plays legally and
  competitively — the search layer adds win-rate on top of a solid heuristic base. This is the
  natural "simplified model": one deck list + one heuristic policy, no beam search, sub-millisecond
  per decision.
- Within the evaluation function, a **small subset of features** — `win`, `no_active`, `prize`,
  `opp_dmg`, `opp_threat` — captures the large majority of decision quality; the remaining features
  are refinements.

## A8. Execution time

- **No training phase** (search-based agent) → training time is effectively zero.
- **Per-decision (inference):** ~**0.12 s** think budget on CPU; the safety fallback is
  sub-millisecond.
- **Per game:** ~1.5–2.9 s in local self-play.
- **Evaluation runs (for tuning):** a 1000-deck × 5-policy field sweep takes on the order of a
  couple of hours on a single CPU; a 60-game paired head-to-head takes ~2–3 minutes.
- **Simplified (rules-only) agent:** negligible per-decision cost.

## A9. References

- Competition environment and card/engine data (`cg` package) provided on the Kaggle Competition
  Website (Competition Data — used only for this competition per the rules).
- Standard game-AI techniques: **beam search**, **determinization for imperfect-information games**
  (sampling hidden state), and **linear state-evaluation heuristics**.

---

## Reproduction (maps to Submission-Model guidelines B1–B8)

- **Entry point:** `main.py` exposes `def agent(obs_dict: dict) -> list[int]` (deck IDs on the first
  call; option indices thereafter).
- **Deck asset:** `deck.csv` (60 card IDs), loaded relative to `main.py` so it resolves correctly at
  the Kaggle runtime path `/kaggle_simulations/agent/`.
- **Engine:** the bundled `cg/` package (CPU-only).
- **Package command:** `python tools/package_submission.py --name submission` →
  `artifacts/submission.tar.gz` (flat top level: `main.py`, `deck.csv`, `agent/`, `cg/`,
  `cards.json`).
- **Environment:** Python 3.13, CPU-only, no GPU, no network access at inference (compliant with the
  "No Ingress or Egress" rule).
- **Determinism / assumptions:** the agent must never raise (an exception is a forfeit); the engine
  shuffles are unseeded, so evaluation is averaged over many games.
- **License:** MIT (per the competition Winner License).
