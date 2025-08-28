# Qudits

A simulation library for quantum computing using qudits {n-dimensional qubits (quantum digits)}.

As I learn from 'Introduction to Quantum Computing: From a Layperson to a Programmer in 30 Steps' by Hiu Yung Wong, I will be implementing the code exercise in the book while understanding the underlying mathematics and extending it beyond qubits to qudits.

I believe implementation of qudits is more viable using photonic quantum computing given the beautiful similarities between wave nature of light and quantum computing.

## Latest Update: Completed first 20 chapters (first 12 chapters are part 1 and just the prereq math with quantum flavour, part 2 is quantum computing).

## Current Objective:
### Phase 1:
- To be able to define d-dimensional generalized qudits and create corresponding hilbert space.
- To be able to implement generalized gates (primary gates and other possible combinations).
- To be able to implement a qudit circuit. 

## Log:
-  Implemented mathematical operations from the first 10 chapters of the book using python.

## Questions:
-  Can we Parallelize Grover's algorithm using photonic high dimensional Qudits to reach the lower bound and speed up each pass using photonics? (using frequency encoding)

## Long term goals:
- Currently implementing GPU acceleration using torch, but plan to eventually shift to native GPU backend.
- Add noise simulation.
- Integrade Quantum Machine Learning Models (reference QML by Peter Wittek)
- Integrate photonics to enable photonic quantum computing for qudits (reference strawberryfields by PennyLane)
- Include Neuromorphic Computing Models (reference snntorch)
- Include finance specific applications such as those in qiskit finance.

  
- Integrate all to create a Neuromorphic Photonic Quantum Machine Learning Library, supporting (or maybe needing) qudits ?

## Licensing:
I do plan to eventually license it, but thats on the horizon for now.

Possible file structure (for personal reference):
```
qudit_sim/
├── qudit_sim/
│   ├── __init__.py
│   ├── gates/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── x_gate.py
│   │   └── custom_gates.py
│   ├── circuits/
│   │   ├── __init__.py
│   │   ├── circuit.py
│   │   └── builder.py
│   ├── states/
│   │   ├── __init__.py
│   │   └── statevector.py
│   ├── simulators/
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   └── statevector_simulator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── math_utils.py
│   │   └── decorators.py
│   └── config.py
├── tests/
│   ├── test_circuits.py
│   ├── test_gates.py
│   └── ...
├── examples/
│   └── teleportation_qudit.py
├── README.md
├── pyproject.toml
└── setup.cfg

```
