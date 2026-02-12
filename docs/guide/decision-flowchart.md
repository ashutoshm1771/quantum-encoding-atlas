# Decision Flowchart

A visual guide to selecting a quantum encoding. Start at the top and follow the arrows — every one of the 16 encodings is reachable.

!!! info "This flowchart mirrors `EncodingDecisionTree.decide()`"
    The flowchart below is a 1:1 representation of the deterministic decision tree implemented in `encoding_atlas.guide.decision_tree`. For the scored recommendation engine (which returns alternatives and confidence), see [Recommendation Architecture](recommendation-architecture.md).

---

## The Flowchart

```mermaid
flowchart TD
    START([Start]) --> L1{"<b>Level 1</b><br/>What is your data type?"}

    L1 -- "binary / discrete" --> BASIS([Basis Encoding])
    L1 -- "continuous" --> L2{"<b>Level 2</b><br/>Does your data have<br/>a known symmetry?"}

    L2 -- "rotation<br/>(exactly 2 features)" --> SO2([SO2 Equivariant])
    L2 -- "cyclic" --> CYCLIC([Cyclic Equivariant])
    L2 -- "permutation_pairs<br/>(even features)" --> SWAP([Swap Equivariant])
    L2 -- "general<br/>(heuristic)" --> SYMI([Symmetry-Inspired])
    L2 -- "none" --> L3{"<b>Level 3</b><br/>Do you want trainable<br/>encoding parameters?"}

    L3 -- "yes" --> TRAIN([Trainable Encoding])
    L3 -- "no" --> L4{"<b>Level 4</b><br/>What is the<br/>problem structure?"}

    L4 -- "combinatorial / graph" --> QAOA([QAOA-Inspired])
    L4 -- "physics simulation" --> HAM([Hamiltonian Encoding])
    L4 -- "time series / periodic" --> DRU_TS([Data Re-uploading])
    L4 -- "none / general" --> L5{"<b>Level 5</b><br/>Do you need specific<br/>feature interactions?"}

    L5 -- "polynomial<br/>(no entanglement)" --> HOA([Higher-Order Angle])
    L5 -- "custom Pauli strings" --> PFM([Pauli Feature Map])
    L5 -- "none" --> L6{"<b>Level 6</b><br/>What is your<br/>optimisation priority?"}

    L6 -- "speed" --> ANGLE([Angle Encoding])
    L6 -- "noise_resilience" --> HWE([Hardware Efficient])
    L6 -- "trainability" --> DRU_PR([Data Re-uploading])
    L6 -- "accuracy" --> L7{"<b>Level 7</b><br/>How many features?"}

    L7 -- "≤ 4" --> IQP([IQP Encoding])
    L7 -- "5 – 8" --> ZZ([ZZ Feature Map])
    L7 -- "> 8" --> AMP([Amplitude Encoding])

    classDef green  fill:#388e3c,stroke:#1b5e20,color:#fff
    classDef blue   fill:#1976d2,stroke:#0d47a1,color:#fff
    classDef purple fill:#7b1fa2,stroke:#4a148c,color:#fff
    classDef yellow fill:#f9a825,stroke:#f57f17,color:#fff
    classDef orange fill:#f57c00,stroke:#e65100,color:#fff
    classDef brown  fill:#5d4037,stroke:#3e2723,color:#fff
    classDef pink   fill:#c2185b,stroke:#880e4f,color:#fff

    class BASIS,ANGLE,HWE green
    class SO2,CYCLIC,SWAP,SYMI blue
    class TRAIN purple
    class QAOA,HAM yellow
    class DRU_TS,DRU_PR orange
    class HOA,PFM brown
    class IQP,ZZ,AMP pink
```

!!! note "Data Re-uploading has two paths"
    `data_reuploading` is reachable via **two** distinct routes: `problem_structure="time_series"` (Level 4) and `priority="trainability"` (Level 6). All other encodings have exactly one path.

---

## Colour Legend

| Colour | Category | Encodings |
|:-------|:---------|:----------|
| :green_square:{ .md-color } Green | Simple / shallow | Basis, Angle, Hardware Efficient |
| :blue_square:{ .md-color } Blue | Symmetry-aware | SO2, Cyclic, Swap Equivariant, Symmetry-Inspired |
| :purple_square:{ .md-color } Purple | Learnable | Trainable |
| :yellow_square:{ .md-color } Yellow | Domain-specific | QAOA-Inspired, Hamiltonian |
| :orange_square:{ .md-color } Orange | Deep / expressive | Data Re-uploading |
| :brown_square:{ .md-color } Brown | Feature interaction | Higher-Order Angle, Pauli Feature Map |
| :red_square:{ .md-color } Pink | Accuracy / kernel | IQP, ZZ Feature Map, Amplitude |

---

## Quick Reference Table

| Level | Condition | Encoding | Key Tradeoff |
|:------|:----------|:---------|:-------------|
| 1 | Binary / discrete data | Basis | Simple but limited to discrete inputs |
| 2 | Rotational symmetry (2 features) | SO2 Equivariant | Rigorous 2D rotation equivariance |
| 2 | Cyclic symmetry | Cyclic Equivariant | Ring-topology Z_n shift symmetry |
| 2 | Permutation pairs (even features) | Swap Equivariant | Rigorous S_2 pair-swap symmetry |
| 2 | General symmetry | Symmetry-Inspired | Heuristic symmetry inductive bias |
| 3 | Trainable parameters desired | Trainable | Learnable but needs optimisation budget |
| 4 | Combinatorial / graph problem | QAOA-Inspired | Cost-mixer structure for optimisation |
| 4 | Physics simulation | Hamiltonian | Trotterised time evolution, deep circuits |
| 4 | Time series / periodic data | Data Re-uploading | Universal approximation, deep circuits |
| 5 | Polynomial interactions (no entanglement) | Higher-Order Angle | Order-k products, classically simulable |
| 5 | Custom Pauli string rotations | Pauli Feature Map | Flexible but complex to configure |
| 6 | Priority = speed | Angle | O(1) depth, no entanglement, no barren plateaus |
| 6 | Priority = noise resilience | Hardware Efficient | Native gates, low CNOT count |
| 6 | Priority = trainability | Data Re-uploading | Universal approximation via repeated encoding |
| 7 | Priority = accuracy, ≤ 4 features | IQP | Provably hard kernels, full entanglement |
| 7 | Priority = accuracy, 5–8 features | ZZ Feature Map | Standard pairwise feature interactions |
| 7 | Priority = accuracy, > 8 features | Amplitude | Exponential compression, logarithmic qubits |

---

## Programmatic Usage

The flowchart above corresponds exactly to `EncodingDecisionTree.decide()`:

```python
from encoding_atlas.guide import EncodingDecisionTree

tree = EncodingDecisionTree()

# Example: continuous data, no symmetry, accuracy priority, 6 features
result = tree.decide(
    data_type="continuous",
    n_features=6,
    priority="accuracy",
)
print(result)  # "zz_feature_map"
```

For ranked recommendations with confidence scores and alternatives, use the scored recommender instead:

```python
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(
    n_features=6,
    priority="accuracy",
    hardware="simulator",
)
print(rec.encoding_name)   # Top pick
print(rec.alternatives)    # Up to 3 runners-up
print(rec.confidence)      # 0.50 – 0.95
```

!!! tip "Want the full details?"
    See [Recommendation Architecture](recommendation-architecture.md) for the complete scoring breakdown, weight hierarchy, confidence mapping, and worked examples.
