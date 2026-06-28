# API Reference

Complete programmatic interface for the Quantum Encoding Atlas. All public classes and functions are documented below with their signatures, parameters, and return types.

---

## Encodings

All encodings inherit from `BaseEncoding` and share a unified interface.

### Base Class

::: encoding_atlas.core.base.BaseEncoding
    options:
      show_root_heading: true
      members:
        - n_qubits
        - depth
        - properties
        - config
        - get_circuit

### Properties

::: encoding_atlas.core.properties.EncodingProperties
    options:
      show_root_heading: true

---

## Encoding Classes

### Angle Encoding

::: encoding_atlas.AngleEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Amplitude Encoding

::: encoding_atlas.AmplitudeEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Basis Encoding

::: encoding_atlas.BasisEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### IQP Encoding

::: encoding_atlas.IQPEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### ZZ Feature Map

::: encoding_atlas.ZZFeatureMap
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Pauli Feature Map

::: encoding_atlas.PauliFeatureMap
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Data Re-uploading

::: encoding_atlas.DataReuploading
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

### Hardware Efficient Encoding

::: encoding_atlas.HardwareEfficientEncoding
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__

---

## Analysis Module

::: encoding_atlas.analysis
    options:
      show_root_heading: true
      show_source: false

---

## Guide Module

::: encoding_atlas.guide.recommender.recommend_encoding
    options:
      show_root_heading: true

::: encoding_atlas.guide.recommender.Recommendation
    options:
      show_root_heading: true

::: encoding_atlas.guide.rules.get_matching_encodings
    options:
      show_root_heading: true

---

## Atlas Module

The empirical benchmark results — measured circuit resources, simulability,
expressibility, entanglement, trainability, noise resilience, and downstream
VQC / quantum-kernel accuracy for all 16 encodings — bundled with the package
as a queryable, read-only API.

::: encoding_atlas.atlas.get_encoding_profile
    options:
      show_root_heading: true

::: encoding_atlas.atlas.rank_encodings
    options:
      show_root_heading: true

::: encoding_atlas.atlas.pareto_front
    options:
      show_root_heading: true

::: encoding_atlas.atlas.hypothesis_verdicts
    options:
      show_root_heading: true

::: encoding_atlas.atlas.atlas_metadata
    options:
      show_root_heading: true

::: encoding_atlas.atlas.EncodingProfile
    options:
      show_root_heading: true

---

## Exceptions

::: encoding_atlas.core.exceptions
    options:
      show_root_heading: true
      show_source: false
