import numpy as np
from typing import List, Union
from utils import basis_state, normalize_state, probability_distribution, decimal_to_base

class SingleQuditState:
    """
    Represents the state of a single qudit (d-dimensional quantum system).
    """
    
    def __init__(self, dimension: int, amplitudes: np.ndarray = None):
        """
        Initialize a single qudit state.

        Args:
            dimension: Dimension of the qudit (d)
            amplitudes: Complex amplitudes for each basis state. If None, initializes to |0⟩
        """
        self.dimension = dimension

        if amplitudes is None:
            # Initialize to |0⟩ state
            self.amplitudes = basis_state(0, dimension)
        else:
            if len(amplitudes) != dimension:
                raise ValueError(f"Amplitudes must have length {dimension}")
            self.amplitudes = np.array(amplitudes, dtype=complex)
            # Normalize the state
            self.amplitudes = normalize_state(self.amplitudes)

    
    def __str__(self) -> str:
        """String representation of the state."""
        state_str = "|ψ⟩ = "
        terms = []
        for i in range(self.dimension):
            amp = self.amplitudes[i]
            if abs(amp) > 1e-10:  # Only show non-zero terms
                terms.append(f"({amp.real:.3f}{amp.imag:+.3f}i)|{i}⟩")
        
        if not terms:
            terms.append("0")
        
        return state_str + " + ".join(terms)
    
    def get_amplitudes(self) -> np.ndarray:
        """Get the state vector amplitudes."""
        return self.amplitudes.copy()
    
    def set_amplitudes(self, new_amplitudes: Union[List[complex], np.ndarray]):
        """Set new amplitudes and normalize."""
        if len(new_amplitudes) != self.dimension:
            raise ValueError(f"Amplitudes must have length {self.dimension}")
        self.amplitudes = normalize_state(np.array(new_amplitudes, dtype=complex))
    
    def probability(self, basis_state_index: int) -> float:
        """
        Compute the probability of measuring a specific basis state.
        
        Args:
            basis_state_index: Index of the basis state (0 to dimension-1)
            
        Returns:
            Probability of measuring that state
        """
        if basis_state_index < 0 or basis_state_index >= self.dimension:
            raise ValueError(f"Basis state index must be between 0 and {self.dimension-1}")
        
        return abs(self.amplitudes[basis_state_index]) ** 2
    
    def probabilities(self) -> np.ndarray:
        """
        Compute probabilities for all basis states.
        
        Returns:
            Array of probabilities for each basis state
        """
        return probability_distribution(self.amplitudes)
    
    def measure(self) -> int:
        """
        Measure the qudit in the computational basis.
        
        Returns:
            Measurement outcome (0 to dimension-1)
        """
        probs = self.probabilities()
        outcome = np.random.choice(self.dimension, p=probs)
        
        # Collapse the state to the measured outcome
        self.amplitudes = basis_state(outcome, self.dimension)
        
        return outcome
    
    def expectation_value(self, operator: np.ndarray) -> float:
        """
        Compute the expectation value of an operator.
        
        Args:
            operator: Hermitian operator as a matrix
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        if operator.shape != (self.dimension, self.dimension):
            raise ValueError(f"Operator must be {self.dimension}x{self.dimension}")
        
        # Compute ⟨ψ|O|ψ⟩
        O_psi = operator @ self.amplitudes
        expectation = np.vdot(self.amplitudes, O_psi)
        
        return np.real(expectation)  # Expectation value should be real for Hermitian operators
    
    def density_matrix(self) -> np.ndarray:
        """
        Compute the density matrix representation.
        
        Returns:
            Density matrix ρ = |ψ⟩⟨ψ|
        """
        return np.outer(self.amplitudes, self.amplitudes.conj())

class MultiQuditState:
    """
    Represents the state of multiple qudits.
    """
    
    def __init__(self, dimensions, amplitudes= None):
        """
        Initialize a multi-qudit state.
        
        Args:
            dimensions: Either a single integer (all qudits same dimension) 
                       or list of dimensions for each qudit
            amplitudes: Complex amplitudes for the full state. If None, initializes to |00...0⟩
        """
        if isinstance(dimensions, int):
            self.dimensions = [dimensions]
            self.num_qudits = 1
        else:
            self.dimensions = dimensions
            self.num_qudits = len(dimensions)
        
        self.total_dimension = np.prod(self.dimensions)
        
        if amplitudes is None:
            # Initialize to |00...0⟩ state
            self.amplitudes = basis_state(0, self.total_dimension)
        else:
            if len(amplitudes) != self.total_dimension:
                raise ValueError(f"Amplitudes must have length {self.total_dimension}")
            self.amplitudes = np.array(amplitudes, dtype=complex)
            # Normalize the state
            self.amplitudes = normalize_state(self.amplitudes)
    
    def __str__(self) -> str:
        """String representation of the state."""
        state_str = "|ψ⟩ = "
        terms = []
        
        for i in range(self.total_dimension):
            amp = self.amplitudes[i]
            if abs(amp) > 1e-10:  # Only show non-zero terms
                # Convert index to multi-qudit state representation
                state_label = self._index_to_state_label(i)
                terms.append(f"({amp.real:.3f}{amp.imag:+.3f}i)|{state_label}⟩")
        
        if not terms:
            terms.append("0")
        
        return state_str + " + ".join(terms)
    
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

    def _labels_to_state_index(self, labels: List[int]) -> int:
        """
        Convert qudit state labels to state vector index in little-endian.
        """
        if len(labels) != self.num_qudits:
            raise ValueError("Number of labels must match number of qudits")
        
        index = 0
        multiplier = 1
        for i, (label, dim) in enumerate(zip(labels, self.dimensions)):
            if label >= dim:
                raise ValueError(f"Label {label} exceeds dimension {dim} for qudit {i}")
            index += label * multiplier
            multiplier *= dim
        return index

    def print_state_detailed(self, state_vector: np.ndarray):
        """Print state vector with proper qudit labels."""
        print("State Vector (little-endian |q1 q0⟩ where q0 is LSB):")
        for i, amp in enumerate(state_vector):
            if abs(amp) > 1e-10:
                labels = self._state_index_to_labels(i)
                label_str = "|" + "".join(str(l) for l in reversed(labels)) + "⟩"  # Reverse for display
                prob = np.abs(amp)**2
            print(f"  {label_str} (index {i}): amp={amp.real:+.3f}{amp.imag:+.3f}j, prob={prob:.3f}")

    '''def _index_to_state_label(self, index: int) -> str:
        """Convert a global index to multi-qudit state label."""
        digits = decimal_to_base(index, max(self.dimensions), self.num_qudits)
        # Truncate to actual dimensions
        label = ""
        for i, digit in enumerate(digits[-self.num_qudits:]):
            label += str(digit)
        return label
    
    def _state_label_to_index(self, label: str) -> int:
        """Convert a multi-qudit state label to global index."""
        if len(label) != self.num_qudits:
            raise ValueError(f"State label must have length {self.num_qudits}")
        
        digits = [int(char) for char in label]
        index = 0
        for i, digit in enumerate(digits):
            if digit >= self.dimensions[i]:
                raise ValueError(f"Digit {digit} exceeds dimension {self.dimensions[i]} for qudit {i}")
            index = index * self.dimensions[i] + digit
        
        return index
    '''
    def get_amplitudes(self) -> np.ndarray:
        """Get the full state vector amplitudes."""
        return self.amplitudes.copy()
    
    def set_amplitudes(self, new_amplitudes: Union[List[complex], np.ndarray]):
        """Set new amplitudes and normalize."""
        if len(new_amplitudes) != self.total_dimension:
            raise ValueError(f"Amplitudes must have length {self.total_dimension}")
        self.amplitudes = normalize_state(np.array(new_amplitudes, dtype=complex))
    
    def probability(self, state_label: str) -> float:
        """
        Compute the probability of measuring a specific multi-qudit basis state.
        
        Args:
            state_label: String representing the basis state (e.g., "01" for |01⟩)
            
        Returns:
            Probability of measuring that state
        """
        index = self._state_label_to_index(state_label)
        return abs(self.amplitudes[index]) ** 2
    
    def probabilities(self) -> np.ndarray:
        """
        Compute probabilities for all basis states.
        
        Returns:
            Array of probabilities for each basis state
        """
        return probability_distribution(self.amplitudes)
    
    def measure_all(self) -> str:
        """
        Measure all qudits in the computational basis.
        
        Returns:
            String representing the measurement outcome (e.g., "012")
        """
        probs = self.probabilities()
        outcome_index = np.random.choice(self.total_dimension, p=probs)
        
        # Collapse the state to the measured outcome
        self.amplitudes = basis_state(outcome_index, self.total_dimension)
        
        # Convert index to state label
        return self._index_to_state_label(outcome_index)
    
    def measure_single(self, qudit_index: int) -> int:
        """
        Measure a single qudit in the computational basis.
        
        Args:
            qudit_index: Index of the qudit to measure (0 to num_qudits-1)
            
        Returns:
            Measurement outcome for that qudit
        """
        if qudit_index < 0 or qudit_index >= self.num_qudits:
            raise ValueError(f"Qudit index must be between 0 and {self.num_qudits-1}")
        
        # Calculate marginal probabilities for the target qudit
        marginal_probs = np.zeros(self.dimensions[qudit_index])
        
        for global_index in range(self.total_dimension):
            # Extract the state of the target qudit
            temp_index = global_index
            for i in range(self.num_qudits-1, -1, -1):
                dim = self.dimensions[i]
                if i == qudit_index:
                    qudit_state = temp_index % dim
                    break
                temp_index //= dim
            
            marginal_probs[qudit_state] += abs(self.amplitudes[global_index]) ** 2
        
        # Normalize (should be already normalized, but just in case)
        marginal_probs /= np.sum(marginal_probs)
        
        # Sample from the marginal distribution
        outcome = np.random.choice(self.dimensions[qudit_index], p=marginal_probs)
        
        # Collapse the state based on the measurement outcome
        new_amplitudes = np.zeros(self.total_dimension, dtype=complex)
        
        for global_index in range(self.total_dimension):
            # Extract the state of the target qudit
            temp_index = global_index
            for i in range(self.num_qudits-1, -1, -1):
                dim = self.dimensions[i]
                if i == qudit_index:
                    qudit_state = temp_index % dim
                    break
                temp_index //= dim
            
            if qudit_state == outcome:
                new_amplitudes[global_index] = self.amplitudes[global_index]
        
        # Normalize the new state
        norm = np.linalg.norm(new_amplitudes)
        if norm > 0:
            new_amplitudes /= norm
            self.amplitudes = new_amplitudes
        
        return outcome
    
    def expectation_value(self, operator: np.ndarray) -> float:
        """
        Compute the expectation value of an operator.
        
        Args:
            operator: Hermitian operator as a matrix
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        if operator.shape != (self.total_dimension, self.total_dimension):
            raise ValueError(f"Operator must be {self.total_dimension}x{self.total_dimension}")
        
        # Compute ⟨ψ|O|ψ⟩
        O_psi = operator @ self.amplitudes
        expectation = np.vdot(self.amplitudes, O_psi)
        
        return np.real(expectation)
    
    def density_matrix(self) -> np.ndarray:
        """
        Compute the density matrix representation.
        
        Returns:
            Density matrix ρ = |ψ⟩⟨ψ|
        """
        return np.outer(self.amplitudes, self.amplitudes.conj())
    
    def partial_trace(self, keep_qudits: List[int]) -> np.ndarray:
        """
        Compute the partial trace over specified qudits.
        
        Args:
            keep_qudits: List of qudit indices to keep
            
        Returns:
            Reduced density matrix for the kept qudits
        """
        from utils import partial_trace
        return partial_trace(self.density_matrix(), self.dimensions, keep_qudits)

# Convenience functions for creating common states
def zero_state(dimension: int) -> SingleQuditState:
    """Create a |0⟩ state for a single qudit."""
    return SingleQuditState(dimension)

def uniform_superposition(dimension: int) -> SingleQuditState:
    """Create a uniform superposition state for a single qudit."""
    amplitudes = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    return SingleQuditState(dimension, amplitudes)

def bell_state(dimension: int = 2) -> MultiQuditState:
    """
    Create a Bell-like state for two qudits of same dimension.
    
    For qubits: (|00⟩ + |11⟩)/√2
    For qutrits: (|00⟩ + |11⟩ + |22⟩)/√3, etc.
    """
    if dimension < 2:
        raise ValueError("Dimension must be at least 2")
    
    total_dim = dimension * dimension
    amplitudes = np.zeros(total_dim, dtype=complex)
    
    for i in range(dimension):
        amplitudes[i * dimension + i] = 1.0
    
    amplitudes /= np.sqrt(dimension)
    return MultiQuditState([dimension, dimension], amplitudes)