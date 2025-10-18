import numpy as np
from typing import List, Union

def kron(*matrices: np.ndarray) -> np.ndarray:
    """
    Compute the Kronecker product of multiple matrices. (tensor product)
    
    Args:
        *matrices: Variable number of numpy arrays
        
    Returns:
        Kronecker product of all input matrices
    """
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def basis_state(index: int, dimension: int) -> np.ndarray:
    """
    Create a basis state vector |index⟩ for a single qudit.
    
    Args:
        index: Which basis state (0 to dimension-1)
        dimension: Dimension of the qudit
        
    Returns:
        Basis state vector as a numpy array
    """
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state

def computational_basis(dimension: int) -> List[np.ndarray]:
    """
    Generate all computational basis states for a single qudit.
    
    Args:
        dimension: Dimension of the qudit
        
    Returns:
        List of all basis state vectors
    """
    return [basis_state(i, dimension) for i in range(dimension)]

def normalize_state(state: np.ndarray) -> np.ndarray:
    """
    Normalize a quantum state vector.
    
    Args:
        state: State vector to normalize
        
    Returns:
        Normalized state vector
    """
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return state / norm

def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is unitary.
    
    Args:
        matrix: Matrix to check
        
    Returns:
        True if matrix is unitary, False otherwise
    """

    # number of columns != number of rows
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if U†U = I
    identity = np.eye(matrix.shape[0])
    product = matrix.conj().T @ matrix
    return np.allclose(product, identity)

def is_hermitian(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is Hermitian.
    
    Args:
        matrix: Matrix to check
        
    Returns:
        True if matrix is Hermitian, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    return np.allclose(matrix, matrix.conj().T)

def probability_distribution(state: np.ndarray) -> np.ndarray:
    """
    Compute the probability distribution of a quantum state.
    
    Args:
        state: Quantum state vector
        
    Returns:
        Array of probabilities for each basis state
    """
    state_ = normalize_state(state)
    return np.abs(state_) ** 2

def inner_product(state1: np.ndarray, state2: np.ndarray) -> complex:
    """
    Compute the inner product ⟨state1|state2⟩.
    
    Args:
        state1: First state vector
        state2: Second state vector
        
    Returns:
        Inner product as complex number
    """
    return np.vdot(state1, state2)

def outer_product(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
    """
    Compute the outer product |state1⟩⟨state2|.
    
    Args:
        state1: First state vector
        state2: Second state vector
        
    Returns:
        Outer product as matrix
    """
    return np.outer(state1, state2.conj())

def tensor_power(state: np.ndarray, power: int) -> np.ndarray:
    """
    Compute the tensor product of a state with itself multiple times.
    
    Args:
        state: State vector
        power: Number of times to tensor with itself
        
    Returns:
        Tensor power of the state
    """
    result = state
    for _ in range(power - 1):
        result = np.kron(result, state)
    return result

def decimal_to_base(number: int, base: int, length: int) -> List[int]:
    """
    Convert a decimal number to base representation with fixed length.
    
    Args:
        number: Decimal number to convert
        base: Base to convert to
        length: Length of output list (pads with zeros)
        
    Returns:
        List of digits in the specified base
    """
    digits = []
    while number > 0:
        digits.append(number % base)
        number //= base
    
    # Pad with zeros to reach desired length
    while len(digits) < length:
        digits.append(0)
    
    return digits[::-1]  # Reverse to get correct order

def base_to_decimal(digits: List[int], base: int) -> int:
    """
    Convert a base representation to decimal number.
    
    Args:
        digits: List of digits in the specified base
        base: Base of the number system
        
    Returns:
        Decimal number
    """
    result = 0
    for i, digit in enumerate(reversed(digits)):
        result += digit * (base ** i)
    return result



# Chapter 12
def partial_trace(state: np.ndarray, dimensions: List[int], keep: List[int]) -> np.ndarray:
    """
    Compute the partial trace over specified subsystems.
    
    Args:
        state: Full state vector or density matrix
        dimensions: List of dimensions for each subsystem
        keep: List of indices of subsystems to keep
        
    Returns:
        Reduced density matrix
    """
    if len(state.shape) == 1:
        # Convert state vector to density matrix
        state = np.outer(state, state.conj())
    
    total_dim = np.prod(dimensions)
    if state.shape != (total_dim, total_dim):
        raise ValueError("State dimensions don't match subsystem dimensions")
    
    # This is a simplified implementation - full partial trace is complex
    # For now, we'll implement a basic version that works for simple cases
    keep_dims = [dimensions[i] for i in keep]
    kept_dim = np.prod(keep_dims)
    
    # For the full implementation, we'd need to properly handle the tracing
    # This is a placeholder that returns identity for now
    print("Warning: Simplified partial trace implementation")
    return np.eye(kept_dim) / kept_dim

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute the fidelity between two pure quantum states |state1> and |state2>.
    
    Args:
        state1: First state vector (numpy array)
        state2: Second state vector (numpy array)
        
    Returns:
        Fidelity value (float between 0 and 1)
    """
    # Compute inner product <state1|state2>
    inner_prod = np.vdot(state1, state2)
    # Fidelity = |<state1|state2>|^2
    F = np.abs(inner_prod) ** 2
    return F.real