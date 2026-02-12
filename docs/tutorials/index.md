# Tutorials

Hands-on guides that walk you through common workflows with the Quantum Encoding Atlas, from your first encoding to systematic benchmarking.

---

## Learning Path

| # | Tutorial | What You'll Learn | Prerequisites |
|:-:|:---------|:-----------------|:-------------|
| 1 | [Getting Started](getting-started.md) | Create encodings, inspect properties, generate circuits | Installation |
| 2 | [Comparing Encodings](comparing-encodings.md) | Analyse and compare encodings side by side | Tutorial 1 |
| 3 | [Custom Encodings](custom-encoding.md) | Build your own encoding by subclassing `BaseEncoding` | Tutorial 1 |
| 4 | [Benchmarking](benchmarking.md) | Run systematic experiments across encodings and datasets | Tutorial 2 |
| 5 | [Hardware Considerations](hardware-considerations.md) | Adapt encodings for real quantum hardware constraints | Tutorial 1 |

---

!!! tip "Running the tutorials"
    All tutorials use PennyLane as the default backend. To follow along:

    ```bash
    pip install encoding-atlas[visualization]
    ```

    Code blocks are self-contained — you can copy-paste them directly into a Python script or Jupyter notebook.
