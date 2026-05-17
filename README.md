# Qudits

A simulation library for quantum computing using qudits {n-dimensional qubits (quantum digits)}.

The idea sparked from 'Introduction to Quantum Computing: From a Layperson to a Programmer in 30 Steps' by Hiu Yung Wong, and now that I've started with the implementation I am referring to further research papers. 

Note: Work on this project has currently been paused due to other commitments.

## Latest Update: Completed Utils/Aux module and started gates module (implemented X,H,Z gate for qudits, multi control gates and Toffoli gates next)

Recent literature referenced:
- [✅] https://arxiv.org/pdf/2410.05122  GENERALISED QUANTUM GATES FOR QUDITS AND THEIR APPLICATION IN QUANTUM FOURIER TRANSFORM 
- [   ] https://arxiv.org/pdf/2303.12979 Optimal Synthesis of Multi-Controlled Qudit Gates 
- [   ] https://arxiv.org/pdf/0806.0654  Efficient Toffoli Gates Using Qudits

## Current Objective:
### Phase 1:
- To be able to define d-dimensional generalized qudits and create corresponding hilbert space.
- To be able to implement generalized gates (primary gates and other possible combinations).

## Long term goals:
- Implementing GPU acceleration using torch.
- Add noise simulation.
- Integrade Quantum Machine Learning Models (reference QML by Peter Wittek)
- Integrate photonics to enable photonic quantum computing for qudits (reference strawberryfields by PennyLane)
- Include Neuromorphic Computing Models (reference snntorch)
  
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
