# Recommendation Engine Architecture

This page explains the internal architecture of the `recommend_encoding()` and `EncodingDecisionTree.decide()` systems — how they score, rank, and select encodings for your problem.

!!! tip "Looking for a quick recommendation?"
    If you just want to pick an encoding, see [Which Encoding?](which-encoding.md) or the [Decision Flowchart](decision-flowchart.md). This page is for understanding *how* the engine works under the hood.

---

## System Overview

```mermaid
flowchart LR
    subgraph INPUT["User Input"]
        direction TB
        I1["n_features"]
        I2["n_samples"]
        I3["task"]
        I4["hardware"]
        I5["priority"]
        I6["data_type"]
        I7["symmetry"]
        I8["trainable"]
        I9["problem_structure"]
        I10["feature_interactions"]
    end

    subgraph KB["Knowledge Base"]
        direction TB
        K1["ENCODING_RULES<br/>(16 encodings)"]
        K2["best_for tags"]
        K3["avoid_when tags"]
        K4["hard constraints"]
        K5["metadata"]
    end

    subgraph OUTPUT["Output"]
        direction TB
        O1["Recommendation"]
        O2["encoding_name"]
        O3["explanation"]
        O4["alternatives\[\]"]
        O5["confidence"]
    end

    INPUT --> KB --> OUTPUT
```

---

## Full Pipeline

The recommendation is produced in two phases, preceded by input validation.

```mermaid
flowchart TD
    START([User calls recommend_encoding]) --> INPUT[/Collect all 10 input parameters/]

    INPUT --> VALIDATE["Validate all inputs<br/>(ValueError on bad values)"]
    VALIDATE --> PHASE_A["<b>PHASE A: Hard Filter</b><br/>Structural elimination"]

    PHASE_A --> LOOP{For each of<br/>16 encodings}
    LOOP --> HC1{data_type in<br/>requires_data_type?}
    HC1 -- "Fails" --> ELIMINATE[Eliminate encoding]
    HC1 -- "Passes / N/A" --> HC2{n_features ==<br/>requires_n_features?}
    HC2 -- "Fails" --> ELIMINATE
    HC2 -- "Passes / N/A" --> HC3{n_features even?<br/>if requires_even}
    HC3 -- "Fails" --> ELIMINATE
    HC3 -- "Passes / N/A" --> HC4{n_features <=<br/>max_features?}
    HC4 -- "Fails" --> ELIMINATE
    HC4 -- "Passes / N/A" --> HC5{symmetry ==<br/>requires_symmetry?}
    HC5 -- "Fails" --> ELIMINATE
    HC5 -- "Passes / N/A" --> HC6{trainable == True?<br/>if requires_trainable}
    HC6 -- "Fails" --> ELIMINATE
    HC6 -- "Passes" --> SURVIVE[Add to candidate pool]

    ELIMINATE --> NEXT_ENC{More encodings?}
    SURVIVE --> NEXT_ENC
    NEXT_ENC -- "Yes" --> LOOP
    NEXT_ENC -- "No" --> EMPTY_CHECK{Candidate pool<br/>empty?}

    EMPTY_CHECK -- "Yes" --> FALLBACK["Return angle encoding<br/>confidence = 0.50<br/>(safe general fallback)"]
    EMPTY_CHECK -- "No" --> PHASE_B["<b>PHASE B: Soft Scoring</b><br/>Score each candidate 0.0 → 1.0<br/>(9 scoring categories)"]

    PHASE_B --> SCORE["Compute weighted score<br/>(see scoring breakdown)"]
    SCORE --> RANK["Sort candidates by score<br/>descending"]
    RANK --> SELECT["Best = rank #1<br/>Alternatives = ranks #2–#4"]
    SELECT --> CONFIDENCE["Map raw score → confidence<br/>(0.50 – 0.95)"]
    CONFIDENCE --> EXPLAIN["Generate explanation<br/>from template"]
    EXPLAIN --> OUTPUT([Return Recommendation])
```

---

## Phase A: Hard Constraint Filter

Each constraint is a **binary gate**. If any single gate fails, the encoding is eliminated regardless of everything else.

```mermaid
flowchart LR
    subgraph "Hard Constraint Chain (ALL must pass)"
        direction LR
        A["1. Data Type<br/>─────────<br/>basis requires<br/>binary/discrete"] --> B["2. Exact Features<br/>─────────<br/>SO2 requires<br/>exactly 2"]
        B --> C["3. Even Features<br/>─────────<br/>swap_equivariant<br/>requires even"]
        C --> D["4. Max Features<br/>─────────<br/>IQP caps at 12<br/>HOAngle caps at 10"]
        D --> E["5. Symmetry<br/>─────────<br/>cyclic_eq requires<br/>symmetry='cyclic'"]
        E --> F["6. Trainable<br/>─────────<br/>trainable enc<br/>requires opt-in"]
    end

    F -- "All pass" --> PASS([CANDIDATE])
    A -- "Any fail" --> FAIL([ELIMINATED])
    B -- "Any fail" --> FAIL
    C -- "Any fail" --> FAIL
    D -- "Any fail" --> FAIL
    E -- "Any fail" --> FAIL
```

### Hard Constraints by Encoding

| Encoding | requires_data_type | requires_n_features | requires_even | max_features | requires_symmetry | requires_trainable |
|---|---|---|---|---|---|---|
| angle | -- | -- | -- | -- | -- | -- |
| basis | binary, discrete | -- | -- | -- | -- | -- |
| higher_order_angle | -- | -- | -- | 10 | -- | -- |
| iqp | -- | -- | -- | 12 | -- | -- |
| zz_feature_map | -- | -- | -- | 12 | -- | -- |
| pauli_feature_map | -- | -- | -- | 12 | -- | -- |
| data_reuploading | -- | -- | -- | 8 | -- | -- |
| hardware_efficient | -- | -- | -- | -- | -- | -- |
| amplitude | -- | -- | -- | -- | -- | -- |
| qaoa | -- | -- | -- | -- | -- | -- |
| hamiltonian | -- | -- | -- | -- | -- | -- |
| trainable | -- | -- | -- | -- | -- | YES |
| symmetry_inspired | -- | -- | -- | -- | general | -- |
| so2_equivariant | -- | 2 | -- | 2 | rotation | -- |
| cyclic_equivariant | -- | -- | -- | -- | cyclic | -- |
| swap_equivariant | -- | -- | YES | -- | permutation_pairs | -- |

!!! note
    **"--"** means no constraint (always passes).

---

## Phase B: Soft Scoring

After the hard filter, each surviving encoding gets a score built from **9 scoring categories** applied in sequence. Weights are extracted as named constants (`_W_*`) for tuning.

```mermaid
flowchart TD
    subgraph SCORE["Score Computation (per candidate)"]
        direction TB

        START_SCORE["score = 0.0"] --> S1

        subgraph S1["Step 1 — Hard-Precondition Bonuses (dominant)"]
            direction TB
            DT["+0.50 _W_DATA_TYPE_BONUS<br/>encoding declares a required data_type<br/>AND user's data_type is in that list"]
            SYM["+0.45 _W_SYMMETRY_BONUS<br/>encoding requires a symmetry type<br/>AND user specified that exact type"]
            NF["+0.10 _W_N_FEATURES_BONUS<br/>encoding requires exact n_features<br/>AND user's n_features matches"]
            TR["+0.40 _W_TRAINABLE_BONUS<br/>encoding requires trainable<br/>AND user set trainable=True"]
        end

        S1 --> S2

        subgraph S2["Step 2 — Binary/Discrete Penalty"]
            direction TB
            PEN1["-0.20 _W_BINARY_PENALTY<br/>user has binary/discrete data BUT<br/>encoding has no data_type requirement"]
        end

        S2 --> S3

        subgraph S3["Step 3 — Priority Matching"]
            direction TB
            PR["+0.10 per tag _W_PRIORITY_PER_TAG (max 2 tags)<br/>user's priority mapped to tags via _PRIORITY_TAG_MAP<br/>each tag found in best_for adds +0.10"]
        end

        S3 --> S4

        subgraph S4["Step 4 — Problem Structure"]
            direction TB
            PS["+0.36 _W_STRUCTURE_BONUS<br/>user's problem_structure mapped to tags<br/>any tag found in best_for"]
        end

        S4 --> S5

        subgraph S5["Step 5 — Feature Interactions"]
            direction TB
            FI["+0.35 _W_INTERACTION_BONUS<br/>polynomial → 'polynomial_features' in best_for<br/>custom_pauli → 'custom_pauli' in best_for"]
        end

        S5 --> S6

        subgraph S6["Step 6 — Task Matching"]
            direction TB
            TM["+0.04 _W_TASK_BONUS<br/>classification → 'kernel_methods' in best_for<br/>regression → 'universal_approximation' in best_for<br/><i>Only when problem_structure AND<br/>feature_interactions are both None</i>"]
        end

        S6 --> S7

        subgraph S7["Step 7 — Hardware Suitability"]
            direction TB
            HW1["+0.10 _W_HARDWARE_NISQ_BONUS<br/>hardware != simulator AND<br/>encoding has nisq/native/noise tags in best_for"]
            HW2["-0.15 _W_HARDWARE_DEEP_PENALTY<br/>hardware != simulator AND<br/>circuit_depth == 'deep'"]
            HW3["-0.08 _W_AVOID_WHEN_PENALTY<br/>hardware != simulator AND<br/>encoding has 'noisy_hardware' or 'nisq_hardware'<br/>in avoid_when tags"]
        end

        S7 --> S8

        subgraph S8["Step 8 — Feature Count"]
            direction TB
            LOG["+0.15 _W_LOGARITHMIC_BONUS<br/>n_features > 8 AND<br/>qubit_scaling == 'logarithmic'"]
            ACC["+0.12 _W_ACCURACY_DEFAULT_BONUS<br/>priority == 'accuracy' AND encoding is<br/>the default for this feature range"]
            SMALL["+0.03 _W_SMALL_FEATURE_BONUS<br/>n_features <= 4 AND<br/>max_features >= n_features"]
        end

        S8 --> S9

        subgraph S9["Step 9 — Sample Count"]
            direction TB
            TINY["+0.03 _W_SMALL_SAMPLE_BONUS<br/>n_samples < 100 AND<br/>simulable == True"]
        end

        S9 --> CLAMP["Clamp to 0.0 – 1.0"]
    end
```

### Weight Hierarchy

```
Step 1 bonuses  (0.10 – 0.50)  ██████████████████████████  Structural match — "this encoding EXISTS for your use case"
Step 2 penalty  (-0.20)        ████████                    Binary mismatch — "wrong data type assumption"
Step 3 bonuses  (0.10 – 0.20)  ██████████                  Priority match — "this encoding FITS your priorities"
Step 4–5 bonuses(0.35 – 0.36)  ██████████████████          Domain match — "this encoding is DESIGNED for your domain"
Step 6 bonus    (0.04)         ██                          Task match — "small nudge for task-relevant encodings"
Step 7 mixed    (-0.15 – +0.10)████████                    Hardware — "NISQ bonus / deep penalty / avoid_when penalty"
Step 8 bonuses  (0.03 – 0.15)  ██████                      Feature count — "qubit efficiency tiebreakers"
Step 9 bonus    (0.03)         ██                          Sample count — "prefer cheap for tiny datasets"
```

!!! info "Key Insight"
    A **single Step 1 match** (+0.40 to +0.50) will always outrank **all Step 3--9 bonuses combined**. This ensures that specialized encodings reliably beat general-purpose ones when the user's constraints match.

---

## Tag Mappings

The user's string parameters are expanded into `best_for` tags for matching.

### Priority Tags

```mermaid
flowchart LR
    subgraph "User Priority → Matched Tags"
        SPEED["priority = 'speed'"] --> SPEED_TAGS["speed, simplicity"]
        NOISE["priority = 'noise_resilience'"] --> NOISE_TAGS["nisq_hardware, native_gates,<br/>noise_resilience"]
        TRAIN["priority = 'trainability'"] --> TRAIN_TAGS["trainability, task_specific,<br/>optimization"]
        ACC["priority = 'accuracy'"] --> ACC_TAGS["expressibility, quantum_advantage,<br/>universal_approximation,<br/>kernel_methods"]
    end
```

### Problem Structure Tags

```mermaid
flowchart LR
    subgraph "Problem Structure → Matched Tags"
        COMB["'combinatorial'"] --> COMB_TAGS["combinatorial,<br/>graph_optimization,<br/>qaoa_structure"]
        PHYS["'physics_simulation'"] --> PHYS_TAGS["physics_simulation,<br/>time_evolution"]
        TS["'time_series'"] --> TS_TAGS["periodic_data,<br/>cyclic_symmetry,<br/>time_series"]
    end
```

### Task Tags

Applied in Step 6, only when no domain-specific parameter (`problem_structure`, `feature_interactions`) is active.

```mermaid
flowchart LR
    subgraph "Task Type → Matched Tags (via _TASK_TAG_MAP)"
        CLS["task = 'classification'"] --> CLS_TAGS["kernel_methods"]
        REG["task = 'regression'"] --> REG_TAGS["universal_approximation"]
    end
```

!!! warning
    Task matching is **suppressed** when `problem_structure` or `feature_interactions` is specified, to prevent interference with higher-priority domain signals.

---

## Confidence Mapping

Raw scores are mapped to a human-interpretable confidence via a **continuous piecewise-linear function** with three bands:

```
Raw Score         Confidence        Band
─────────         ──────────        ────
>= 0.50    ───→   0.85 – 0.95      HIGH       "Strong structural match"
0.30–0.49  ───→   0.65 – 0.85      MEDIUM     "Good priority/structure match"
< 0.30     ───→   0.50 – 0.65      LOWER      "Weak match / fallback"

Fallback   ───→   0.50             MINIMAL    "No candidates survived hard filter"
                                               (= _score_to_confidence(0.0))
```

The function is **continuous at both band boundaries**: score=0.30 → 0.65, score=0.50 → 0.85.

```mermaid
flowchart LR
    SCORE["Raw Score"] --> CHECK1{">= 0.50?"}
    CHECK1 -- "Yes" --> HIGH["0.85 + (score - 0.50) * 0.20<br/>capped at 0.95"]
    CHECK1 -- "No" --> CHECK2{">= 0.30?"}
    CHECK2 -- "Yes" --> MED["0.65 + (score - 0.30) * 1.00"]
    CHECK2 -- "No" --> LOW["0.50 + score * 0.50"]
```

---

## Decision Tree (Alternative Path)

`EncodingDecisionTree.decide()` provides a **deterministic single-answer** alternative to the scored recommender. It walks a fixed priority chain -- the first matching level returns immediately:

```mermaid
flowchart TD
    START([decide]) --> L1{"<b>Level 1</b><br/>data_type?"}

    L1 -- "binary / discrete" --> BASIS([basis])
    L1 -- "continuous" --> L2{"<b>Level 2</b><br/>symmetry?"}

    L2 -- "rotation<br/>(n_features=2)" --> SO2([so2_equivariant])
    L2 -- "cyclic" --> CYC([cyclic_equivariant])
    L2 -- "permutation_pairs<br/>(even features)" --> SWAP([swap_equivariant])
    L2 -- "general" --> SYMI([symmetry_inspired])
    L2 -- "none" --> L3{"<b>Level 3</b><br/>trainable?"}

    L3 -- "yes" --> TRAINABLE([trainable])
    L3 -- "no" --> L4{"<b>Level 4</b><br/>problem_structure?"}

    L4 -- "combinatorial" --> QAOA([qaoa])
    L4 -- "physics_simulation" --> HAM([hamiltonian])
    L4 -- "time_series" --> DRU2([data_reuploading])
    L4 -- "none" --> L5{"<b>Level 5</b><br/>feature_interactions?"}

    L5 -- "polynomial" --> HOA([higher_order_angle])
    L5 -- "custom_pauli" --> PFM([pauli_feature_map])
    L5 -- "none" --> L6{"<b>Level 6</b><br/>priority?"}

    L6 -- "speed" --> ANGLE([angle])
    L6 -- "noise_resilience" --> HWE([hardware_efficient])
    L6 -- "trainability" --> DRU([data_reuploading])
    L6 -- "accuracy" --> L7{"<b>Level 7</b><br/>n_features?"}

    L7 -- "<= 4" --> IQP([iqp])
    L7 -- "5 – 8" --> ZZ([zz_feature_map])
    L7 -- "> 8" --> AMP([amplitude])

    style BASIS fill:#4a9,stroke:#333,color:#fff
    style SO2 fill:#4a9,stroke:#333,color:#fff
    style CYC fill:#4a9,stroke:#333,color:#fff
    style SWAP fill:#4a9,stroke:#333,color:#fff
    style SYMI fill:#4a9,stroke:#333,color:#fff
    style TRAINABLE fill:#4a9,stroke:#333,color:#fff
    style QAOA fill:#4a9,stroke:#333,color:#fff
    style HAM fill:#4a9,stroke:#333,color:#fff
    style HOA fill:#4a9,stroke:#333,color:#fff
    style PFM fill:#4a9,stroke:#333,color:#fff
    style ANGLE fill:#4a9,stroke:#333,color:#fff
    style HWE fill:#4a9,stroke:#333,color:#fff
    style DRU fill:#4a9,stroke:#333,color:#fff
    style DRU2 fill:#4a9,stroke:#333,color:#fff
    style IQP fill:#4a9,stroke:#333,color:#fff
    style ZZ fill:#4a9,stroke:#333,color:#fff
    style AMP fill:#4a9,stroke:#333,color:#fff
```

!!! note
    All 16 encodings are reachable. `data_reuploading` is reachable by **two** distinct paths: `problem_structure="time_series"` (Level 4) and `priority="trainability"` (Level 6).

---

## Recommender vs Decision Tree

| Aspect | `recommend_encoding()` | `EncodingDecisionTree.decide()` |
|---|---|---|
| **Output** | Top pick + 3 alternatives + confidence + explanation | Single encoding name |
| **Method** | Hard filter → soft scoring → ranking | Fixed priority-level walk |
| **Nuance** | Weighs multiple signals simultaneously | First match wins |
| **Confidence** | Returns 0.50--0.95 confidence score | No confidence (deterministic) |
| **Fallback** | Returns `angle` with confidence 0.50 | Always returns something |
| **Validation** | `ValueError` on invalid inputs | `ValueError` on invalid inputs |
| **Use case** | "Explore trade-offs" | "Just tell me what to use" |
| **Consistency** | Top pick usually agrees with decision tree | Always deterministic |

---

## Worked Example

**Input:**

```python
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(
    n_features=6,
    priority="accuracy",
    hardware="ibm",
    data_type="continuous",
)
```

### Phase A -- Hard Filter

```
angle               — no constraints          → SURVIVES
basis               — requires binary/discrete → ELIMINATED (continuous)
higher_order_angle  — max_features=10          → SURVIVES (6 <= 10)
iqp                 — max_features=12          → SURVIVES (6 <= 12)
zz_feature_map      — max_features=12          → SURVIVES
pauli_feature_map   — max_features=12          → SURVIVES
data_reuploading    — max_features=8           → SURVIVES (6 <= 8)
hardware_efficient  — no constraints           → SURVIVES
amplitude           — no constraints           → SURVIVES
qaoa                — no constraints           → SURVIVES
hamiltonian         — no constraints           → SURVIVES
trainable           — requires trainable=True  → ELIMINATED (trainable=False)
symmetry_inspired   — requires symmetry=general→ ELIMINATED (symmetry=None)
so2_equivariant     — requires symmetry=rotation→ ELIMINATED
cyclic_equivariant  — requires symmetry=cyclic → ELIMINATED
swap_equivariant    — requires symmetry=perm.  → ELIMINATED

Candidates: 11 survive
```

### Phase B -- Scoring (top candidates)

```
Encoding            Step 1   Step 2  Step 3       Step 6     Step 7           Step 8       Total
                    (precon) (bin)   (priority)   (task)     (hardware)       (feat. cnt)
──────────          ──────── ──────  ──────────   ──────     ────────────     ───────────  ─────
iqp                 +0.00    —       +0.20        +0.04      -0.08 (avoid)    +0.03        ≈ 0.19
                                     (2 acc tags) (classif)  (noisy_hw+nisq)  (small feat)
zz_feature_map      +0.00    —       +0.10        +0.04      +0.00            +0.12        ≈ 0.26
                                     (kernel)     (classif)                   (acc default)
pauli_feature_map   +0.00    —       +0.10        +0.00      +0.00            +0.03        ≈ 0.13
                                     (kernel)                                 (small feat)
data_reuploading    +0.00    —       +0.10        +0.00      -0.15 (deep)     +0.00        ≈ 0.00
                                     (univ.)                  -0.08 (avoid)
hardware_efficient  +0.00    —       +0.00        +0.00      +0.10 (nisq)     +0.00        ≈ 0.10
amplitude           +0.00    —       +0.00        +0.00      -0.15 (deep)     +0.00        ≈ 0.00
                                                              -0.08 (avoid)
angle               +0.00    —       +0.00        +0.00      +0.00            +0.00        ≈ 0.00
hamiltonian         +0.00    —       +0.00        +0.00      -0.15 (deep)     +0.00        ≈ 0.00
                                                              -0.08 (avoid)
```

### Result

```python
Recommendation(
    encoding_name  = "zz_feature_map",
    explanation    = "ZZ Feature Map provides standard pairwise feature interactions ...",
    alternatives   = ["iqp", "pauli_feature_map", "hardware_efficient"],
    confidence     = 0.63,    # lower band — weak soft match only
)
```

!!! example "Why the low confidence?"
    No Step 1 bonuses fired because the user didn't specify symmetry, data type constraints, or trainable. The system is saying: *"ZZ Feature Map is my best guess, but I'm not very confident because your requirements are generic."*

    The `avoid_when` penalty (Step 7) penalises IQP on real hardware (`"noisy_hardware"` and `"nisq_hardware"` in its `avoid_when` tags), allowing ZZ Feature Map to overtake it despite IQP having more priority tag matches.
