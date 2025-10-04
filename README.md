# Qudits

A simulation library for quantum computing using qudits {n-dimensional qubits (quantum digits)}.

The idea sparked from 'Introduction to Quantum Computing: From a Layperson to a Programmer in 30 Steps' by Hiu Yung Wong, and now that I've started with the implementation I am referring to further research papers. 

I believe implementation of qudits is more viable using photonic quantum computing given the beautiful similarities between wave nature of light and quantum computing.

## Latest Update: Completed Utils/Aux module and started gates module (implemented X,H,Z gate for qudits, multi control gates and Toffoli gates next up)

## Current Objective:
### Phase 1:
- To be able to define d-dimensional generalized qudits and create corresponding hilbert space.
- To be able to implement generalized gates (primary gates and other possible combinations).
- To be able to implement a qudit circuit. 

## Questions:
-  Can we Parallelize Grover's algorithm using photonic high dimensional Qudits to reach the lower bound and speed up each pass using photonics? (using frequency encoding)

## Long term goals:
- Implementing GPU acceleration using torch, but plan to eventually shift to native GPU backend.
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
