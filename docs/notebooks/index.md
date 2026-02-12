# Interactive Notebooks

Hands-on Jupyter notebooks for every encoding in the Quantum Encoding Atlas.
Each notebook is a self-contained guide that walks you through **creation,
circuit generation, analysis, and a complete ML workflow** using the library.

!!! tip "Running locally"
    Download any notebook and run it in your own environment:

    ```bash
    pip install encoding-atlas[all]
    jupyter lab
    ```

---

## Encoding Notebooks

| Encoding | Notebook | Highlights |
|----------|----------|------------|
| **Amplitude** | [amplitude_encoding.ipynb](amplitude_encoding.ipynb) | Logarithmic qubit scaling, normalization, cross-backend verification |
| **Angle** | [angle_encoding.ipynb](angle_encoding.ipynb) | Rotation axis comparison (RX/RY/RZ), product states, baseline benchmarking |
| **Basis** | [basis_encoding.ipynb](basis_encoding.ipynb) | Binary encoding, Clifford simulability, minimal depth |
| **IQP** | [iqp_encoding.ipynb](iqp_encoding.ipynb) | Entanglement topologies, provable classical hardness, expressibility |
| **ZZ Feature Map** | [zz_feature_map.ipynb](zz_feature_map.ipynb) | Pairwise interactions, Pauli-ZZ entanglement, feature correlations |
| **Pauli Feature Map** | [pauli_feature_map.ipynb](pauli_feature_map.ipynb) | Configurable Pauli operators, multi-body interactions |
| **Data Re-uploading** | [data_reuploading.ipynb](data_reuploading.ipynb) | Repeated data injection, universal approximation, trainable layers |
| **Hardware Efficient** | [hardware_efficient_encoding.ipynb](hardware_efficient_encoding.ipynb) | Native gate sets, device-aware topology, NISQ optimization |
| **Higher-Order Angle** | [higher_order_angle_encoding.ipynb](higher_order_angle_encoding.ipynb) | Polynomial feature maps, non-linear transformations |
| **QAOA-Inspired** | [qaoa_encoding.ipynb](qaoa_encoding.ipynb) | Mixer/cost layer structure, combinatorial encoding |
| **Hamiltonian** | [hamiltonian_encoding.ipynb](hamiltonian_encoding.ipynb) | IQP/Pauli-Z/XY/Heisenberg types, evolution time, Trotter steps |
| **Trainable** | [trainable_encoding.ipynb](trainable_encoding.ipynb) | Parameter management, initialization strategies, variational training loop |
| **Symmetry-Inspired** | [symmetry_inspired_feature_map.ipynb](symmetry_inspired_feature_map.ipynb) | Symmetry-preserving circuits, structured feature maps |
| **SO(2) Equivariant** | [so2_equivariant_feature_map.ipynb](so2_equivariant_feature_map.ipynb) | Continuous rotational symmetry, equivariance verification |
| **Cyclic Equivariant** | [cyclic_equivariant_feature_map.ipynb](cyclic_equivariant_feature_map.ipynb) | Discrete cyclic symmetry, group actions, statistical verification |
| **Swap Equivariant** | [swap_equivariant_encoding.ipynb](swap_equivariant_encoding.ipynb) | Permutation symmetry, particle-exchange invariance |

---

## What each notebook covers

Every notebook follows a consistent structure:

1. **Setup & instantiation** -- creating the encoding with different configurations
2. **Core properties** -- inspecting qubits, depth, gate counts, and lazy-loaded properties
3. **Circuit generation** -- building circuits for PennyLane, Qiskit, and Cirq
4. **Batch processing** -- encoding multiple data samples efficiently
5. **Analysis tools** -- simulability, expressibility, entanglement, and trainability
6. **Advanced features** -- registry, serialization, thread safety, protocols
7. **Complete workflow** -- end-to-end quantum kernel or variational classifier example
