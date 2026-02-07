# Data Re-uploading Encoding

**Author:** Ashutosh Mishra

A universal quantum encoding that achieves **arbitrary function approximation**
by repeatedly encoding classical data throughout the circuit, interleaved with
entangling layers — the quantum analogue of depth in classical neural networks.

```
 ╔══════════════════════════════════════════════════════════════════════╗
 ║                                                                      ║
 ║   |ψ(x)⟩  =  [ U_ent · U_data(x) ]^L  |0⟩^⊗n                      ║
 ║                                                                      ║
 ║   "Re-upload the data  →  build a Fourier series  →  universality"  ║
 ║                                                                      ║
 ╚══════════════════════════════════════════════════════════════════════╝
```

---

## 1. The Core Idea

A single layer of rotation gates can only produce simple sinusoidal
functions of the input. By **re-encoding the same data** at every layer,
each repetition adds a new Fourier harmonic — and with enough layers,
the circuit can approximate **any continuous function**.

```
                   Single-layer encoding             Data re-uploading (L layers)
                   ─────────────────────             ───────────────────────────

  Expressivity:    f(x) = a₀ + a₁cos(x)             f(x) = Σₖ₌₀ᴸ [aₖcos(kx)
                          + b₁sin(x)                        + bₖsin(kx)]

  Fourier terms:   1 harmonic                         L harmonics

  Analogy:         Single neuron                      Deep neural network
```

```
  Classical Data          Layer 1              Layer 2            Layer L
  ┌──────────┐       ┌────────────────┐   ┌────────────────┐       ┌────┐
  │ x₀       │───┬──►│ RY(x₀) + CNOT │──►│ RY(x₀) + CNOT │──►···►│    │──► |ψ(x)⟩
  │ x₁       │───┤   └────────────────┘   └────────────────┘       └────┘
  │ x₂       │───┤
  │ ...       │───┘   Same data x re-uploaded at every layer
  └──────────┘
                      ◄───────── Universal approximation ──────────►
```

> The same input vector **x** is fed into every layer. This is the crucial
> difference from single-pass encodings. Each re-upload enriches the
> circuit's Fourier spectrum, enabling it to fit progressively more
> complex functions.

---

## 2. Circuit Structure

Each of the `L` layers consists of two sublayers:
1. **Data encoding** — RY(xᵢ) rotations mapping features to qubits
2. **Entangling** — CNOT ladder creating inter-qubit correlations

### Full Circuit for 4 Qubits, 3 Layers

```
            ┌── Layer 1 ──┐  ┌── Layer 2 ──┐  ┌── Layer 3 ──┐

  |0⟩ ─ RY(x₀) ─■───────── RY(x₀) ─■───────── RY(x₀) ─■──────────
                 │                   │                   │
  |0⟩ ─ RY(x₁) ─X──■────── RY(x₁) ─X──■────── RY(x₁) ─X──■───────
                    │                   │                   │
  |0⟩ ─ RY(x₂) ────X──■─── RY(x₂) ────X──■─── RY(x₂) ────X──■────
                       │                   │                   │
  |0⟩ ─ RY(x₃) ───────X─── RY(x₃) ───────X─── RY(x₃) ───────X────

          ▲         ▲          ▲         ▲          ▲         ▲
       Encoding  Entangling  Encoding Entangling  Encoding Entangling
```

### Anatomy of One Layer

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                         SINGLE LAYER                            │
  │                                                                 │
  │   ┌──────────────────────┐    ┌───────────────────────────┐    │
  │   │   Data Encoding      │    │   Entangling (CNOT        │    │
  │   │   Sublayer            │    │   Ladder)                 │    │
  │   │                      │    │                           │    │
  │   │   q₀ ← RY(x₀)      │    │   q₀ ──■──               │    │
  │   │   q₁ ← RY(x₁)      │    │   q₁ ──X──■──            │    │
  │   │   q₂ ← RY(x₂)      │    │   q₂ ─────X──■──         │    │
  │   │   q₃ ← RY(x₃)      │    │   q₃ ────────X──         │    │
  │   │                      │    │                           │    │
  │   │  (cyclic mapping:    │    │  Linear nearest-neighbor  │    │
  │   │   feature i → qubit  │    │  connectivity only        │    │
  │   │   i mod n_qubits)    │    │                           │    │
  │   └──────────────────────┘    └───────────────────────────┘    │
  │                                                                 │
  │   Depth contribution:  ceil(n_features / n_qubits) + (n-1)    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 3. Why RY Gates — Not RX, Not RZ

Data re-uploading uses **RY rotations exclusively**. This is a deliberate
design choice rooted in how each gate acts on |0⟩:

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │                      GATE COMPARISON ON |0⟩                           │
  ├──────────┬────────────────────────────┬──────────────────────────────┤
  │          │                            │                              │
  │  RY(θ)   │  cos(θ/2)|0⟩ + sin(θ/2)|1⟩│  Real amplitudes ★          │
  │          │                            │  Creates superposition       │
  │          │                            │  Sweeps from |0⟩ to |1⟩     │
  │          │                            │                              │
  ├──────────┼────────────────────────────┼──────────────────────────────┤
  │          │                            │                              │
  │  RX(θ)   │  cos(θ/2)|0⟩              │  Complex amplitudes          │
  │          │  - i·sin(θ/2)|1⟩          │  Introduces imaginary part   │
  │          │                            │  Creates superposition       │
  │          │                            │                              │
  ├──────────┼────────────────────────────┼──────────────────────────────┤
  │          │                            │                              │
  │  RZ(θ)   │  e^{-iθ/2} |0⟩           │  Global phase only!          │
  │          │                            │  NO superposition created    │
  │          │                            │  Useless on |0⟩ alone       │
  │          │                            │                              │
  └──────────┴────────────────────────────┴──────────────────────────────┘
```

```
  Bloch Sphere Trajectories from |0⟩ (north pole):

                    |0⟩
                     *          RY(θ): Sweeps down the X-Z great circle
                    /|\              (real amplitudes, most intuitive)
                   / | \
           RX ←── /  |  \ ──► RY    RX(θ): Sweeps down the Y-Z great circle
                 /   |   \          (complex amplitudes)
                ────-+-────
                     |              RZ(θ): Stays at |0⟩ (!)
                     |              (only adds unmeasurable global phase)
                     *
                    |1⟩
```

**Why RY is the right choice for data re-uploading:**

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  1. No Hadamard needed — RY creates superposition directly from     │
  │     |0⟩. Unlike PauliFeatureMap/ZZFeatureMap which need H gates     │
  │     before RZ rotations, data re-uploading skips that overhead.     │
  │                                                                      │
  │  2. Real amplitudes — keeps the math clean and interpretable.       │
  │     Measurement probabilities are simple cos²/sin² functions.       │
  │                                                                      │
  │  3. Full population transfer — RY(π)|0⟩ = |1⟩, so the full         │
  │     dynamic range of the qubit is accessible for any x ∈ [0, 2π].  │
  │                                                                      │
  │  4. Composability — consecutive RY rotations on the same qubit      │
  │     compose naturally: RY(a)·RY(b) rotates further along the same   │
  │     great circle, which is exactly what cyclic mapping exploits.    │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

> **Contrast with PauliFeatureMap**: PauliFeatureMap applies H gates first
> to create |+⟩, then uses RZ to encode phases. Data re-uploading needs
> no initial superposition layer — RY does both jobs (superposition +
> encoding) in a single gate.

---

## 4. The CNOT Ladder — How Entanglement Is Built

The entangling sublayer uses a **linear CNOT ladder**: sequential CNOT
gates from each qubit to its neighbor, cascading down the register.

```
  CNOT Ladder (4 qubits):

  q₀ ──■────────────
       │
  q₁ ──X──■─────────
          │
  q₂ ─────X──■──────
             │
  q₃ ────────X──────

  3 CNOTs for 4 qubits  (always n_qubits - 1 CNOTs)
```

### What the Ladder Does to a Product State

```
  Before CNOT ladder (product state after RY encoding):

    q₀ = α₀|0⟩ + β₀|1⟩
    q₁ = α₁|0⟩ + β₁|1⟩
    q₂ = α₂|0⟩ + β₂|1⟩

  After CNOT(q₀, q₁):
    q₀ and q₁ become entangled — their states are now correlated
    If q₀ was in |1⟩, q₁ gets flipped

  After CNOT(q₁, q₂):
    Entanglement propagates — q₂ is now correlated with q₁ AND q₀
    Information flows down the chain

  Result: All qubits are connected through a chain of correlations.

  ┌──────────────────────────────────────────────────────────────────┐
  │  The ladder creates NEAREST-NEIGHBOR entanglement that          │
  │  propagates transitively through the chain:                     │
  │                                                                  │
  │    q₀ ↔ q₁ ↔ q₂ ↔ q₃                                         │
  │                                                                  │
  │  After L layers, information from q₀ can reach q₃ even though  │
  │  they never directly interact. Repeated layers build up global  │
  │  correlations from local connections.                           │
  └──────────────────────────────────────────────────────────────────┘
```

### Why Linear Ladder (Not Full Entanglement)

```
  ┌─────────────────────────────┐  ┌──────────────────────────────┐
  │  CNOT Ladder (this encoding)│  │  Full Entanglement           │
  ├─────────────────────────────┤  ├──────────────────────────────┤
  │                             │  │                              │
  │  q₀ ──■────────            │  │  q₀ ──■──■──■──             │
  │       │                    │  │       │  │  │               │
  │  q₁ ──X──■─────            │  │  q₁ ──X──│──│──■──■──      │
  │          │                 │  │          │  │  │  │         │
  │  q₂ ─────X──■──            │  │  q₂ ─────X──│──X──│──■──   │
  │             │              │  │             │     │  │      │
  │  q₃ ────────X──            │  │  q₃ ────────X────X──X──    │
  │                             │  │                              │
  │  CNOTs: n-1 = 3            │  │  CNOTs: n(n-1)/2 = 6        │
  │  Connectivity: linear      │  │  Connectivity: all-to-all   │
  │                             │  │                              │
  └─────────────────────────────┘  └──────────────────────────────┘

  Data re-uploading deliberately uses the LADDER because:

  1. Hardware-friendly: Only nearest-neighbor connections needed
     (maps directly to superconducting qubit chains)

  2. Depth is compensated by REPETITION: after L layers of ladders,
     even distant qubits become strongly correlated

  3. Linear gate count: O(n) per layer vs. O(n²) for full

  4. The universal approximation theorem doesn't require full
     connectivity — depth (re-uploading) provides the expressivity
```

---

## 5. The Fourier Perspective — Why Re-uploading Works

This is the key theoretical insight from Schuld, Sweke & Meyer (2021)
and Perez-Salinas et al. (2020).

### Single Qubit, Single Layer

A single RY rotation produces a quantum state whose measurement
expectation values are degree-1 trigonometric polynomials:

```
  RY(x)|0⟩ = cos(x/2)|0⟩ + sin(x/2)|1⟩

  ⟨Z⟩ = cos(x)     ← only 1 Fourier frequency
```

### Single Qubit, L Layers (Re-uploading)

Composing L rotations (with intervening processing) creates a
**truncated Fourier series** with L+1 terms:

```
  Layer 1        Layer 2           Layer L
  RY(x) ──── W₁ ──── RY(x) ──── W₂ ──── ... ──── RY(x) ──── W_L
              ▲               ▲                            ▲
              │               │                            │
         CNOT ladder     CNOT ladder                  CNOT ladder
        (entangling)    (entangling)                 (entangling)

  The Wₖ matrices ARE the CNOT ladders — they are the "processing"
  between data uploads that enables the Fourier expansion.

  ⟨O⟩ = a₀ + Σₖ₌₁ᴸ [aₖ cos(kx) + bₖ sin(kx)]

  Each layer adds one new Fourier harmonic:

  L=1:  ~~~~~~~~               1 frequency     (simple sine)
  L=2:  ~~~~∿∿∿∿               2 frequencies   (sum of 2 sines)
  L=3:  ~~∿∿≋≋≋≋               3 frequencies   (richer function)
  L=5:  ~∿≋⩪⩪⩪⩪⩪⩪             5 frequencies   (complex functions)
```

### Multi-Qubit with Entanglement

With n qubits and entangling gates, the Fourier spectrum becomes
**multi-dimensional** with up to 2^n frequency components:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Single qubit, L layers:     L+1  Fourier frequencies      │
  │   n qubits, no entanglement:  n × (L+1)  frequencies        │
  │   n qubits + entanglement:    up to (L+1)^n  frequencies    │
  │                                                              │
  │   Entanglement creates CROSS-TERMS between qubits,          │
  │   exponentially expanding the function space.                │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

### Visual: Function Approximation vs. Number of Layers

```
  Target function: ─── (some complex f(x))

  L=1 layer:
        ╱╲
       ╱  ╲        Poor fit — only 1 harmonic
  ────╱    ╲────

  L=3 layers:
       ╱╲ ╱╲
      ╱  ╳  ╲      Better — captures main features
  ───╱  ╱ ╲  ╲───

  L=10 layers:
      ╱╲╱╲╱╲╱╲
     ╱╲╱╲╱╲╱╲╱╲    Excellent — closely matches target
  ──╱╲╱╲╱╲╱╲╱╲╱──

  Accuracy improves with layers  →  Universal Approximation
```

---

## 6. No Hadamard Layer — A Key Structural Difference

Unlike many other encodings (PauliFeatureMap, ZZFeatureMap, IQP), data
re-uploading does **not** use an initial Hadamard layer. This is because
RY does both jobs at once.

```
  PauliFeatureMap / ZZFeatureMap:           Data Re-uploading:
  ──────────────────────────────            ──────────────────

  |0⟩ ── H ── RZ(2x) ──                   |0⟩ ── RY(x) ──

  Step 1: H creates |+⟩                   Step 1: RY creates superposition
  Step 2: RZ encodes phase                         AND encodes data
                                                   (in a single gate!)
  Two gates needed to encode               One gate does everything.
  one feature.

  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  Why does PauliFeatureMap need H?                             │
  │  Because RZ(θ)|0⟩ = e^{-iθ/2}|0⟩ — just a global phase!     │
  │  You MUST superpose first with H to make RZ meaningful.       │
  │                                                                │
  │  Why doesn't DataReuploading need H?                          │
  │  Because RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩                │
  │  RY creates its own superposition. No prep step needed.       │
  │                                                                │
  │  Gate savings per layer: n Hadamard gates eliminated.         │
  │  Over L layers: L × n fewer gates than H+RZ designs.         │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

---

## 7. Cyclic Feature Mapping

When `n_features > n_qubits`, features are mapped cyclically:
feature `i` goes to qubit `i mod n_qubits`.

```
  Example: 6 features → 3 qubits

  Feature   Qubit Assignment     RY gates on qubit (per layer)
  ──────────────────────────────────────────────────────────────
  x₀        qubit 0              RY(x₀) then RY(x₃)
  x₁        qubit 1              RY(x₁) then RY(x₄)
  x₂        qubit 2              RY(x₂) then RY(x₅)
  x₃        qubit 0  (recycled)       ↑ stacked on same qubit
  x₄        qubit 1  (recycled)
  x₅        qubit 2  (recycled)

  Circuit for ONE layer:
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  q₀ ── RY(x₀) ── RY(x₃) ──■──                    │
  │                             │                      │
  │  q₁ ── RY(x₁) ── RY(x₄) ──X──■──                 │
  │                                │                   │
  │  q₂ ── RY(x₂) ── RY(x₅) ─────X──                 │
  │                                                     │
  │  Encoding depth per layer: ceil(6/3) = 2           │
  └─────────────────────────────────────────────────────┘
```

### How Stacked Rotations Compose

```
  When two features land on the same qubit:

  RY(x₃) · RY(x₀) |0⟩

  This is NOT simply RY(x₀ + x₃)|0⟩ in general!
  For a qubit already in a superposition, the second rotation
  creates a more complex state than either alone.

  However, for the very first layer on |0⟩ specifically:
    RY(x₃)·RY(x₀)|0⟩ = RY(x₀ + x₃)|0⟩  (same axis, additive)

  In later layers (where the qubit is entangled with others),
  the CNOT-mediated correlations break this simple additivity,
  and each feature contributes INDEPENDENTLY to the Fourier spectrum.

  ┌──────────────────────────────────────────────────────────────┐
  │  Cyclic mapping trades WIDTH for DEPTH:                     │
  │                                                              │
  │    n_qubits = n_features:  Depth per layer = 1 + (n-1)     │
  │    n_qubits < n_features:  Depth per layer = ceil(f/q)+(q-1)│
  │                                                              │
  │  Fewer qubits → deeper encoding sublayer → same total gates │
  └──────────────────────────────────────────────────────────────┘
```

> This allows encoding more features than qubits, trading circuit
> width for depth. However, features sharing a qubit may interfere.

---

## 8. Single-Qubit Universality

A remarkable property: even with **just 1 qubit**, data re-uploading
achieves universal approximation for single-variable functions.

```
  1 qubit, L layers:

  |0⟩ ── RY(x) ── RY(x) ── RY(x) ── ... ── RY(x) ──  (L times)
                                                         ↓
                                              ⟨O⟩ = Fourier series
                                                    with L terms

  No entanglement needed — depth alone provides expressivity.
  This is analogous to a deep 1-neuron network.

  Wait — without entangling gates, isn't this just RY(L·x)?

  In this PURE DATA-ONLY version, yes: consecutive RY gates
  on a single qubit compose additively (same axis).

  The universality result from Pérez-Salinas et al. assumes
  TRAINABLE intermediate rotations between re-uploads:

  |0⟩ ── RY(x) ── U(θ₁) ── RY(x) ── U(θ₂) ── ... ── RY(x) ──

  where U(θₖ) are parameterized processing gates. The learnable
  parameters {θₖ} control the Fourier coefficients {aₖ, bₖ}.

  ┌────────────────────────────────────────────────────────┐
  │  "A single qubit can learn any function of a single   │
  │   variable, given enough re-uploading layers."         │
  │                                                        │
  │   — Pérez-Salinas et al., Quantum 4, 226 (2020)      │
  └────────────────────────────────────────────────────────┘
```

---

## 9. Choosing n_layers — A Decision Guide

The number of layers is the single most important parameter. It directly
controls the encoding's Fourier frequency content and hence its ability
to represent complex functions.

```
  START
    │
    ├── How complex is your target function?
    │     │
    │     ├── Simple (linearly separable, smooth)
    │     │     └── n_layers = 1-2
    │     │
    │     ├── Moderate (nonlinear boundaries, moderate oscillation)
    │     │     └── n_layers = 3-5  ★ (default = 3)
    │     │
    │     └── Complex (highly oscillatory, sharp features)
    │           └── n_layers = 6-10
    │                (watch trainability!)
    │
    ├── How many qubits?
    │     │
    │     ├── n_qubits = 1  → Need more layers (all expressivity from depth)
    │     │     └── n_layers = 5-20
    │     │
    │     ├── n_qubits = 2-4  → Moderate layers
    │     │     └── n_layers = 3-8
    │     │
    │     └── n_qubits > 4  → Entanglement adds expressivity
    │           └── n_layers = 2-5 often sufficient
    │
    └── Hardware constraints?
          │
          ├── Simulator  → No depth limit, go deep
          ├── NISQ device (short coherence)  → n_layers ≤ 5
          └── Error-corrected  → n_layers as needed

  ┌──────────────────────────────────────────────────────────────────┐
  │  RULE OF THUMB                                                   │
  │                                                                  │
  │  Max Fourier frequency = n_layers                               │
  │  If your function has features at frequency k, you need L ≥ k.  │
  │                                                                  │
  │  More qubits + entanglement = exponentially richer function     │
  │  space, so fewer layers needed for the same task complexity.    │
  └──────────────────────────────────────────────────────────────────┘
```

### Quick Reference: Common Configurations

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │  CONFIGURATION            │  L  │ Qubits │  NOTES                    │
  ├───────────────────────────┼─────┼────────┼───────────────────────────┤
  │  Default                  │  3  │  = n_f │  Good general-purpose     │
  │  DataReuploading(n_f)     │     │        │  starting point           │
  ├───────────────────────────┼─────┼────────┼───────────────────────────┤
  │  Shallow / NISQ           │  2  │  = n_f │  Minimal depth, fast      │
  │  DataReuploading(n_f, 2)  │     │        │  to train, low noise      │
  ├───────────────────────────┼─────┼────────┼───────────────────────────┤
  │  Compact                  │  5  │  < n_f │  Fewer qubits, more depth │
  │  DataReuploading(8,5,4)   │     │        │  Width-for-depth trade    │
  ├───────────────────────────┼─────┼────────┼───────────────────────────┤
  │  Deep / expressive        │ 10  │  = n_f │  High Fourier content     │
  │  DataReuploading(n_f, 10) │     │        │  Watch trainability!      │
  ├───────────────────────────┼─────┼────────┼───────────────────────────┤
  │  Single-qubit universal   │ 15+ │   1    │  Proves universality with │
  │  DataReuploading(1, 15)   │     │        │  just one qubit           │
  └───────────────────────────┴─────┴────────┴───────────────────────────┘
```

---

## 10. Key Properties

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                   DATA RE-UPLOADING PROPERTIES                          │
  ├───────────────────────┬─────────────────────────────────────────────────┤
  │  Qubits required      │  n (or fewer with cyclic mapping)              │
  │  Circuit depth        │  L × [ceil(n_feat/n_qub) + (n_qub - 1)]       │
  │  Total gates          │  L × n_features (RY) + L × (n_qubits-1) (CX)  │
  │  Single-qubit gates   │  L × n_features  (RY rotations)                │
  │  Two-qubit gates      │  L × (n_qubits - 1)  (CNOT ladder)            │
  │  Entangling?          │  Yes (when n_qubits > 1)                       │
  │  Universality         │  Yes — universal function approximation        │
  │  Simulability         │  Not classically simulable (entangling case)   │
  │  Trainability         │  Good: ~max(0.4, 0.9 - 0.05·L)                │
  │  Fourier frequencies  │  Up to (L+1)^n with entanglement               │
  │  Connectivity         │  Linear (nearest-neighbor) only                 │
  │  Native gates         │  RY + CNOT  (only 2 gate types!)               │
  │  Initial state        │  |0⟩^⊗n  (no Hadamard prep needed)            │
  └───────────────────────┴─────────────────────────────────────────────────┘
```

### Why These Properties Matter

```
  Expressibility  ██████████████████░░  ~0.85  ← High: universal approximation
  Trainability    ████████████████░░░░  ~0.75  ← Good: interleaved helps gradients
  Hardware Cost   ████████████░░░░░░░░  Medium ← Linear connectivity, moderate depth
  Noise Impact    ██████████░░░░░░░░░░  Medium ← Depth grows with layers
  Gate Simplicity ██████████████████░░  High   ← Only RY + CNOT, no basis changes
```

---

## 11. Comparison: Depth vs. Width Trade-off

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │       Angle Encoding             Data Re-uploading                 │
  │       (width approach)           (depth approach)                  │
  │                                                                     │
  │  |0⟩ ─ RY(x₀) ─────────    |0⟩ ─ RY(x₀) ─■─ RY(x₀) ─■─ ...    │
  │  |0⟩ ─ RY(x₁) ─────────    |0⟩ ─ RY(x₁) ─X─ RY(x₁) ─X─ ...    │
  │  |0⟩ ─ RY(x₂) ─────────                                          │
  │  |0⟩ ─ RY(x₃) ─────────    4 features on 2 qubits (cyclic)      │
  │                                                                     │
  │  4 qubits, depth 1           2 qubits, depth 2L                   │
  │  No entanglement              Entanglement + universality          │
  │  1 Fourier frequency          L Fourier frequencies                │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 12. Comparison with Other Encodings

```
  ┌───────────────────┬────────┬───────────┬────────────┬──────────────┬──────────────┐
  │    Encoding       │ Qubits │   Depth   │ Entangling │ Simulability │ Universality │
  ├───────────────────┼────────┼───────────┼────────────┼──────────────┼──────────────┤
  │  Angle            │   n    │   reps    │     No     │  Simulable   │     No       │
  │  Data Reupload ★  │  ≤ n   │  O(L·n)  │    Yes     │  Not sim.    │    Yes ★     │
  │  IQP              │   n    │  O(n²)   │    Yes     │  Not sim.    │     No       │
  │  ZZ Feature Map   │   n    │  O(n²)   │    Yes     │  Not sim.    │     No       │
  │  Pauli Feat. Map  │   n    │  O(r·d)  │    Yes     │  Not sim.    │     No       │
  │  Amplitude        │ log₂n │  O(2ⁿ)   │    Yes     │  Not sim.    │     No       │
  └───────────────────┴────────┴───────────┴────────────┴──────────────┴──────────────┘

  Data re-uploading is the only encoding in this table with provable
  universal approximation — it can represent any continuous function.
```

### Structural Comparison with PauliFeatureMap

```
  Both are multi-layer encodings, but their architecture differs fundamentally:

  ┌────────────────────────┬───────────────────┬───────────────────────┐
  │                        │ Data Re-uploading │ PauliFeatureMap       │
  ├────────────────────────┼───────────────────┼───────────────────────┤
  │  Initial state         │ |0⟩               │ |0⟩                   │
  │  Superposition from    │ RY (built-in)     │ H gate (explicit)     │
  │  Data encoding gate    │ RY only           │ RX, RY, or RZ        │
  │  Feature interaction   │ Implicit (CNOT    │ Explicit (π-xᵢ)(π-xⱼ)│
  │                        │ + re-uploading)   │ product mapping       │
  │  Entanglement          │ Linear ladder     │ full / linear / ring  │
  │  Universality          │ Yes               │ No                    │
  │  Qubits ≤ features?    │ Yes (cyclic)      │ No (must equal)       │
  │  Gate diversity        │ 2 types (RY, CX)  │ Many (H,RX,RY,RZ,CX) │
  └────────────────────────┴───────────────────┴───────────────────────┘

  DataReuploading: simple gate set, universality through REPETITION
  PauliFeatureMap: rich gate set, expressivity through PAULI DIVERSITY
```

---

## 13. Trainability and Barren Plateaus

The interleaved data-entanglement structure provides better gradient
flow than purely random circuits, but very deep circuits still face
challenges.

```
  Trainability vs. Number of Layers

  1.0 ┤
      │ ████
  0.9 ┤ ████ ████
      │ ████ ████ ████
  0.8 ┤ ████ ████ ████ ████
      │ ████ ████ ████ ████ ████
  0.7 ┤ ████ ████ ████ ████ ████ ████
      │ ████ ████ ████ ████ ████ ████ ████
  0.6 ┤ ████ ████ ████ ████ ████ ████ ████ ████
      │ ████ ████ ████ ████ ████ ████ ████ ████ ████
  0.5 ┤ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
      │ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ▓▓▓▓
  0.4 ┤ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ▓▓▓▓ ▓▓▓▓
      └──┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬───
         1    2    3    4    5    6    7    8    9   10   11   12
                              n_layers                 ▲
                                                       │
                                              Deep circuit warning
                                              threshold (L > 10)

  ████  Good trainability          ▓▓▓▓  Saturated (floor = 0.4)

  Rule of thumb: trainability ≈ max(0.4, 0.9 - 0.05 × L)
```

### Why DataReuploading Trains Better Than Random Circuits

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  Random / hardware-efficient ansatz:                                │
  │                                                                      │
  │    Parameters are everywhere, data appears once.                    │
  │    Gradients vanish exponentially with depth and width.             │
  │    The circuit quickly becomes "noise-like."                        │
  │                                                                      │
  │  Data re-uploading:                                                 │
  │                                                                      │
  │    Data re-appears at every layer, providing "anchor points"        │
  │    that constrain the circuit's behavior.                           │
  │    The interleaved RY(x)─CNOT─RY(x)─CNOT structure prevents the   │
  │    loss landscape from becoming completely flat because the         │
  │    data injection breaks the symmetry that causes barren plateaus.  │
  │                                                                      │
  │    Analogy: skip connections in ResNets prevent vanishing           │
  │    gradients by re-injecting the input. Data re-uploading does     │
  │    the same thing by re-injecting the data.                        │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

### Mitigation Strategies for Deep Circuits (L > 10)

```
  1. Gradient-free optimizers     COBYLA, Nelder-Mead, SPSA
                                  Bypass the gradient problem entirely

  2. Layer-wise training          Train layers 1-3 first, then freeze
                                  and add layers 4-6, etc.
                                  Avoids deep gradient propagation

  3. Parameter initialization     Initialize near identity:
                                  small random angles keep early
                                  gradients meaningful

  4. Reduce n_layers              The default L=3 is chosen precisely
                                  to avoid this regime
```

---

## 14. Data Preprocessing

RY rotations are 2pi-periodic. Features should be scaled to use the
full rotation range meaningfully.

```
  Rotation angle
  ──────────────────────────────────────────────────────────►
  0              pi             2pi            3pi
  |──────────────|──────────────|──────────────|
  |   ← meaningful range →     |  (periodic)  |

  RY(θ)|0⟩ at key angles:

  θ = 0    → |0⟩                  (no rotation)
  θ = π/2  → 0.924|0⟩ + 0.383|1⟩ (partial superposition)
  θ = π    → 0.707|0⟩ + 0.707|1⟩ (equal superposition)
  θ = 2π   → |0⟩                  (full rotation, back to start)

  With re-uploading, the effective range matters even more:
  each layer re-encodes the data, amplifying any scaling issues.
```

### Recommended Scaling

```
  Method                  Formula                        Output Range
  ──────────────────────────────────────────────────────────────────────
  Min-max to [0, 2pi]    x' = 2pi·(x - min)/(max - min)  [0, 2pi]
  Standard + scale       x' = pi·(x - mu) / sigma         ~ [-pi, pi]
  Arctan                 x' = 2·arctan(x)                  (-pi, pi)

  Implementation note: values beyond |4pi| trigger a debug-level
  warning to flag potentially unscaled data. The circuit still works
  (RY is periodic), but extreme values reduce discriminability
  between different inputs.
```

---

## 15. Example Walkthrough

Encode `x = [0.5, 1.2]` on 2 qubits with 2 layers:

```
  Configuration: n_features=2, n_qubits=2, n_layers=2

  Initial state: |00⟩

  ═══════════════════════════════════════════════════════
  LAYER 1
  ═══════════════════════════════════════════════════════

  Step 1a — Data encoding:
  ─────────────────────────
  q₀: RY(0.5)|0⟩  = cos(0.25)|0⟩ + sin(0.25)|1⟩
                   = 0.969|0⟩ + 0.247|1⟩

  q₁: RY(1.2)|0⟩  = cos(0.60)|0⟩ + sin(0.60)|1⟩
                   = 0.825|0⟩ + 0.565|1⟩

  Combined (tensor product — no entanglement yet):
  |ψ⟩ = 0.800|00⟩ + 0.547|01⟩ + 0.204|10⟩ + 0.139|11⟩

  Step 1b — CNOT entangling:
  ───────────────────────────
  CNOT(q₀, q₁):  |00⟩→|00⟩  |01⟩→|01⟩  |10⟩→|11⟩  |11⟩→|10⟩

  |ψ⟩ = 0.800|00⟩ + 0.547|01⟩ + 0.139|10⟩ + 0.204|11⟩
         ↑                       ↑ swapped     ↑ swapped
  The state is now ENTANGLED — it cannot be written as q₀ ⊗ q₁.

  ═══════════════════════════════════════════════════════
  LAYER 2  (same data re-uploaded!)
  ═══════════════════════════════════════════════════════

  Step 2a — Data encoding (same x applied again):
  ─────────────────────────────────────────────────
  RY(0.5) applied to q₀,  RY(1.2) applied to q₁
  Each qubit's local state is further rotated, but
  the entanglement from Layer 1 is preserved.

  This is where re-uploading's power lies: the RY gates
  now act on an ENTANGLED state, creating a richer
  function of x than Layer 1 alone could produce.

  Step 2b — CNOT entangling:
  ───────────────────────────
  CNOT(q₀, q₁) applied again, further correlating the qubits.

  ═══════════════════════════════════════════════════════

  Final state |ψ(x)⟩: An entangled 2-qubit state encoding
  the input through 2 Fourier harmonics, capable of
  representing more complex functions than a single layer.
```

---

## 16. Resource Scaling

```
  n_features │ n_qubits │ L │  RY Gates │ CNOT Gates │  Total │  Depth
  ───────────┼──────────┼───┼───────────┼────────────┼────────┼────────
       2     │    2     │ 3 │      6    │      3     │     9  │    6
       4     │    4     │ 3 │     12    │      9     │    21  │   12
       4     │    4     │ 5 │     20    │     15     │    35  │   20
       8     │    4     │ 3 │     24    │      9     │    33  │   15
       8     │    8     │ 3 │     24    │     21     │    45  │   24
      16     │    8     │ 5 │     80    │     35     │   115  │   45
       1     │    1     │10 │     10    │      0     │    10  │   10

  Scaling formulas:
    RY gates   = L × n_features
    CNOT gates = L × max(0, n_qubits - 1)
    Depth      = L × [ceil(n_features / n_qubits) + (n_qubits - 1)]

  Compare gate scaling with other entangling encodings:
  ┌───────────────────┬────────────────────┬──────────────────────┐
  │   Encoding        │  CNOT scaling      │  For n=8, default    │
  ├───────────────────┼────────────────────┼──────────────────────┤
  │  DataReuploading  │  O(L × n)          │  21 CNOTs  (L=3)     │
  │  ZZFeatureMap     │  O(reps × n²)      │  56 CNOTs  (reps=2)  │
  │  PauliFeatureMap  │  O(reps × n²)      │  56 CNOTs  (reps=2)  │
  │  IQP              │  O(reps × n²)      │  56 CNOTs  (reps=2)  │
  └───────────────────┴────────────────────┴──────────────────────┘
  DataReuploading has the lowest CNOT count among entangling encodings.
```

---

## 17. Strengths and Limitations

```
         STRENGTHS                              LIMITATIONS
  ┌────────────────────────────────┐  ┌────────────────────────────────┐
  │                                │  │                                │
  │  + Universal approximation    │  │  - Depth grows with layers    │
  │    Provably can learn any      │  │    May exceed NISQ coherence   │
  │    continuous function         │  │    times for many layers       │
  │                                │  │                                │
  │  + Single-qubit universality  │  │  - Trainability degrades      │
  │    Works even with 1 qubit     │  │    Deep circuits (L>10) face   │
  │    (unique property!)          │  │    barren plateau saturation   │
  │                                │  │                                │
  │  + Fewer qubits via cyclic    │  │  - Feature interference       │
  │    mapping (depth-for-width)   │  │    Cyclic mapping may cause    │
  │                                │  │    features to conflict        │
  │  + Hardware-friendly          │  │                                │
  │    Linear connectivity only    │  │  - Gate count scaling         │
  │    (nearest-neighbor CNOTs)    │  │    O(L × n) total gates       │
  │                                │  │                                │
  │  + Good gradient flow         │  │  - No proven quantum          │
  │    Interleaved structure       │  │    advantage for all tasks     │
  │    helps training convergence  │  │                                │
  │                                │  │                                │
  │  + Minimal gate set           │  │  - No explicit feature         │
  │    Only RY + CNOT needed       │  │    interactions (unlike ZZ     │
  │    (no H, S, Sdg overhead)     │  │    which encodes xᵢ·xⱼ)      │
  │                                │  │                                │
  │  + No Hadamard overhead       │  │                                │
  │    RY handles superposition    │  │                                │
  │    + encoding in one gate      │  │                                │
  │                                │  │                                │
  └────────────────────────────────┘  └────────────────────────────────┘
```

---

## 18. Use Cases

```
                          Best suited for
                    ┌──────────────────────────────┐
                    │                              │
  ┌─────────────────┤  Quantum Classifiers         │  Universal approximation
  │                 │  (binary & multi-class)       │  enables complex boundaries
  │                 ├──────────────────────────────┤
  │                 │                              │
  ├─────────────────┤  Function Fitting            │  Fourier series matches
  │                 │  (regression tasks)           │  continuous target functions
  │                 ├──────────────────────────────┤
  │                 │                              │
  ├─────────────────┤  Variational Algorithms      │  Input encoding layer for
  │                 │  (VQE, QAOA, custom VQC)     │  parameterized circuits
  │                 ├──────────────────────────────┤
  │                 │                              │
  ├─────────────────┤  Quantum Neural Networks     │  Depth-based expressivity
  │                 │  (QNN architectures)         │  without excessive qubits
  │                 ├──────────────────────────────┤
  │                 │                              │
  ├─────────────────┤  Qubit-Constrained Devices   │  Cyclic mapping encodes
  │                 │  (few-qubit hardware)        │  many features on few qubits
  │                 ├──────────────────────────────┤
  │                 │                              │
  └─────────────────┤  Single-Qubit Experiments    │  Demonstrate QML with
                    │  (educational / prototyping) │  minimal hardware
                    └──────────────────────────────┘
```

---

## 19. Connection to Classical Deep Learning

```
  ┌──────────────────────────┐     ┌──────────────────────────────┐
  │   Classical Deep Network │     │   Data Re-uploading Circuit  │
  ├──────────────────────────┤     ├──────────────────────────────┤
  │                          │     │                              │
  │  Input x ──► Layer 1     │     │  Input x ──► RY(x) + CNOT   │
  │          ──► Layer 2     │     │          ──► RY(x) + CNOT   │
  │          ──► ...         │     │          ──► ...             │
  │          ──► Layer L     │     │          ──► RY(x) + CNOT   │
  │          ──► Output      │     │          ──► Measurement     │
  │                          │     │                              │
  │  Residual connections    │     │  Data re-uploading           │
  │  preserve input info     │     │  preserves input info        │
  │                          │     │                              │
  │  Width × Depth           │     │  2^n Hilbert space × Depth  │
  │  = expressivity          │     │  = expressivity              │
  │                          │     │                              │
  │  Neurons: sigmoid/ReLU   │     │  Qubits: RY (cos/sin)       │
  │  Linear → nonlinear      │     │  Rotation → entanglement     │
  │                          │     │                              │
  └──────────────────────────┘     └──────────────────────────────┘

  Key parallel: each layer sees the raw input again, preventing
  information loss — analogous to skip/residual connections.

  Key difference: the quantum version operates in an exponentially
  large Hilbert space (2^n dimensions), potentially encoding richer
  representations with fewer "neurons" (qubits).
```

---

## References

1. Pérez-Salinas, A., et al. (2020). "Data re-uploading for a universal
   quantum classifier." Quantum, 4, 226.

2. Schuld, M., Sweke, R., & Meyer, J. K. (2021). "Effect of data encoding
   on the expressive power of variational quantum machine learning models."
   Physical Review A, 103(3), 032430.

3. Goto, T., et al. (2021). "Universal approximation property of quantum
   machine learning models in quantum-enhanced feature spaces."
   Physical Review Letters, 127(9), 090506.

4. McClean, J. R., et al. (2018). "Barren plateaus in quantum neural
   network training landscapes." Nature Communications, 9, 4812.