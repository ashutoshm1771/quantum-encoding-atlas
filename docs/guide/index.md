# Decision Guide

Choosing the right quantum encoding for your problem is one of the most consequential decisions in a quantum machine learning pipeline. This guide helps you navigate that choice systematically.

---

## Three Ways to Choose

<div class="grid cards" markdown>

-   **[Which Encoding?](which-encoding.md)**

    ---

    Answer a few questions about your data and priorities, and get a recommendation with reasoning.

-   **[Decision Flowchart](decision-flowchart.md)**

    ---

    A visual flowchart that walks you from problem characteristics to encoding choice in under a minute.

-   **[Recommendation Architecture](recommendation-architecture.md)**

    ---

    Deep dive into how the scoring engine works — hard filters, soft scoring weights, confidence mapping, and worked examples.

-   **[FAQ](faq.md)**

    ---

    Quick answers to the most common questions about encoding selection.

</div>

---

## Programmatic Recommendations

The atlas includes a recommendation engine that you can call directly from Python:

```python
from encoding_atlas.guide import recommend_encoding

rec = recommend_encoding(
    n_features=4,
    n_samples=500,
    task='classification',
    hardware='simulator',
    priority='accuracy'
)

print(rec.encoding_name)   # e.g., 'IQPEncoding'
print(rec.explanation)     # Human-readable reasoning
```

You can also query the rule engine directly:

```python
from encoding_atlas.guide import get_matching_encodings

matches = get_matching_encodings(
    n_features=4,
    is_entangling=True,
    max_depth=50,
)
```

See the [API Reference](../api/index.md) for the full guide interface.

---

## Quick Rules of Thumb

| Your Situation | Start With |
|:--------------|:-----------|
| First time, just exploring | Angle Encoding |
| Need provable quantum advantage | IQP Encoding |
| Many features, few qubits | Amplitude Encoding |
| Data has known symmetry | Equivariant encodings |
| Training a variational model | Data Re-uploading or Trainable |
| Deploying on real hardware | Hardware Efficient or Angle |
| Comparing quantum vs classical | IQP + classical baselines |
