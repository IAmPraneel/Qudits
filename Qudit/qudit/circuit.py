# circuit.py - Core circuit simulation module for qudits
import numpy as np
from typing import List, Dict, Any, Optional
from gates import QuditGates

class QuditCircuit:
    """
    A class to represent and simulate quantum circuits for qudits of arbitrary dimensions.
    
    This module focuses on simulation clarity and follows the gate definitions from
    'Generalised Quantum Gates for Qudits and their Application in Quantum Fourier Transform'.
    """
    
    def __init__(self, num_qudits: int, dimensions: List[int]):
        """
        Initialize a quantum circuit with multiple qudits.
        
        Args:
            num_qudits: Number of quantum registers (qudits) in the circuit
            dimensions: List of dimensions for each qudit [d1, d2, ..., dn]
        """
        if len(dimensions) != num_qudits:
            raise ValueError("Length of dimensions list must match number of qudits")
        
        self.num_qudits = num_qudits
        self.dimensions = dimensions
        self.total_dimension = np.prod(dimensions)
        
        # Initialize to |00...0⟩ state
        self.state = np.zeros(self.total_dimension, dtype=complex)
        self.state[0] = 1.0
        
        # Circuit operations stored as list of (gate, targets, params)
        self.operations = []
        
        # Measurement results
        self.measurement_results = {}
    
    def apply_gate(self, gate_matrix: np.ndarray, target_qudits: List[int]):
        """
        Apply a gate to specific target qudits.
        
        Args:
            gate_matrix: Unitary gate matrix to apply
            target_qudits: List of qudit indices to apply the gate to
        """
        # Validate gate dimensions match target qudit dimensions
        target_dims = [self.dimensions[i] for i in target_qudits]
        expected_dim = np.prod(target_dims)
        
        if gate_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"Gate matrix dimension {gate_matrix.shape} doesn't match "
                           f"target qudit dimensions {expected_dim}")
        
        # Store operation
        self.operations.append(('gate', gate_matrix, target_qudits))
        
        # Apply gate to current state
        full_operator = self._build_full_operator(gate_matrix, target_qudits)
        self.state = full_operator @ self.state
    
    def _build_full_operator(self, gate: np.ndarray, target_qudits: List[int]) -> np.ndarray:
        """
        Build full operator matrix for little-endian system.
        
        In little-endian: |q0, q1, ..., qn⟩ where q0 is LEAST significant
        Tensor product order: I_{n-1} ⊗ I_{n-2} ⊗ ... ⊗ Gate ⊗ ... ⊗ I_0
        where Gate is applied to the target qudit position
        """
        # For single-qudit gates
        if len(target_qudits) == 1:
            target = target_qudits[0]
            
            # Build tensor product from MOST significant to LEAST significant
            # This is the key fix: we need to reverse the order
            operators = []
            for qudit_index in range(self.num_qudits-1, -1, -1):
                if qudit_index == target:
                    operators.append(gate)
                else:
                    operators.append(np.eye(self.dimensions[qudit_index]))
            
            # Now build the tensor product in the correct order
            full_op = operators[0]
            for op in operators[1:]:
                full_op = np.kron(full_op, op)
            
            return full_op
        else:
            # Multi-qudit gates - more complex, handle separately
            raise NotImplementedError("Multi-qudit gates not yet implemented")
    
    def x(self, target: int):
        """Apply X gate to a target qudit."""
        x_gate = QuditGates.x_gate(self.dimensions[target])
        self.apply_gate(x_gate, [target])
        return self
    
    def z(self, target: int):
        """Apply Z gate to a target qudit."""
        z_gate = QuditGates.z_gate(self.dimensions[target])
        self.apply_gate(z_gate, [target])
        return self
    
    def h(self, target: int):
        """Apply Hadamard gate to a target qudit."""
        h_gate = QuditGates.hadamard_gate(self.dimensions[target])
        self.apply_gate(h_gate, [target])
        return self
    
    def measure(self, target: int, shots: int = 1) -> Dict[int, int]:
        """
        Measure a specific qudit in the computational basis.
        
        Args:
            target: Index of qudit to measure
            shots: Number of measurement shots
            
        Returns:
            Dictionary mapping measurement outcomes to counts
        """
        outcomes = []
        
        for _ in range(shots):
            # Calculate probabilities for each state of the target qudit
            probs = self._marginal_probabilities(target)
            
            # Sample from probability distribution
            outcome = np.random.choice(self.dimensions[target], p=probs)
            outcomes.append(outcome)
            
            # Collapse state (in real simulation)
            # This would require state update based on measurement outcome
        
        # Count outcomes
        counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
        self.measurement_results[f'q{target}'] = counts
        
        return counts
    
    def _marginal_probabilities(self, target: int) -> np.ndarray:
        """
        Calculate marginal probabilities for a specific qudit.
        """
        probs = np.zeros(self.dimensions[target])
        
        for i in range(self.total_dimension):
            # Extract the target qudit's state from the global index
            temp_index = i
            for qudit_idx in range(self.num_qudits-1, -1, -1):
                dim = self.dimensions[qudit_idx]
                if qudit_idx == target:
                    target_state = temp_index % dim
                    break
                temp_index //= dim
            
            probs[target_state] += abs(self.state[i]) ** 2
        
        # Normalize
        probs /= np.sum(probs)
        return probs
    
    def get_state(self) -> np.ndarray:
        """Return the current state vector."""
        return self.state.copy()
    
    def set_state(self, new_state: np.ndarray):
        """Set the circuit to a specific state vector."""
        if len(new_state) != self.total_dimension:
            raise ValueError(f"State vector must have length {self.total_dimension}")
        if not np.isclose(np.linalg.norm(new_state), 1.0):
            raise ValueError("State vector must be normalized")
        self.state = new_state.copy()
    
    def __str__(self) -> str:
        """String representation of the circuit."""
        return (f"QuditCircuit({self.num_qudits} qudits, "
                f"dimensions: {self.dimensions}, "
                f"operations: {len(self.operations)})")

    def _state_index_to_labels(self, index: int) -> List[int]:
        """
        Convert state vector index to qudit state labels in little-endian.
        
        In little-endian: |q0, q1, ..., qn⟩ where q0 is LEAST significant
        index = q0 + q1*d0 + q2*d0*d1 + ... 
        """
        labels = []
        temp = index
        for dim in self.dimensions:
            labels.append(temp % dim)
            temp //= dim
        return labels

    def print_state_detailed(self):
        """Print state vector with proper qudit labels."""
        print("State Vector (little-endian |q1 q0⟩ where q0 is LSB):")
        for i, amp in enumerate(self.state):
            if abs(amp) > 1e-10:
                labels = self._state_index_to_labels(i)
                # Reverse for display to show most significant first
                label_str = "|" + "".join(str(l) for l in reversed(labels)) + "⟩"
                prob = np.abs(amp)**2
                print(f"  {label_str} (index {i}): amp={amp.real:+.3f}{amp.imag:+.3f}j, prob={prob:.3f}")
