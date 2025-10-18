# gates.py 
# https://arxiv.org/pdf/2410.05122  GENERALISED QUANTUM GATES FOR QUDITS AND THEIR APPLICATION IN QUANTUM FOURIER TRANSFORM
# https://arxiv.org/pdf/2303.12979 Optimal Synthesis of Multi-Controlled Qudit Gates
# https://arxiv.org/pdf/0806.0654  Efficient Toffoli Gates Using Qudits

import numpy as np
from typing import List

class QuditGates:
    """
    Generalized qudit gates following the formulations from:
    'Generalised Quantum Gates for Qudits and their Application in Quantum Fourier Transform'
    """
    
    @staticmethod
    def get_omega(dimension: int) -> complex:
        """
        Get the fundamental phase factor for dimension d.
        ω = e^(2πi/d) is a primitive d-th root of unity.
        
        Args:
            dimension: Dimension of the qudit (d)
            
        Returns:
            Complex phase factor ω
        """
        return np.exp(2j * np.pi / dimension)
    
    @staticmethod
    def x_gate(dimension: int) -> np.ndarray:
        """
        Generalized Pauli-X gate (Shift operator) for qudits.
        This is a cyclic permutation matrix that shifts basis states.
        
        According to the paper: X_d|j⟩ = |(j+1) mod d⟩
        
        Args:
            dimension: Dimension of the qudit (d)
            
        Returns:
            d × d unitary matrix representing the X gate
        """
        X = np.zeros((dimension, dimension), dtype=complex)
        
        # Create the cyclic shift: each state |j⟩ maps to |(j+1) mod d⟩
        for i in range(dimension):
            X[(i + 1) % dimension, i] = 1.0
            
        return X
    
    @staticmethod
    def z_gate(dimension: int) -> np.ndarray:
        """
        Generalized Pauli-Z gate (Clock operator) for qudits.
        This applies phases to each basis state.
        
        According to the paper: Z_d|j⟩ = ω^j |j⟩ where ω = e^(2πi/d)
        
        Args:
            dimension: Dimension of the qudit (d)
            
        Returns:
            d × d unitary matrix representing the Z gate
        """
        Z = np.zeros((dimension, dimension), dtype=complex)
        omega = QuditGates.get_omega(dimension)
        
        # Apply phase ω^j to each basis state |j⟩
        for i in range(dimension):
            Z[i, i] = omega ** i
            
        return Z
    
    @staticmethod
    def hadamard_gate(dimension: int) -> np.ndarray:
        """
        Generalized Hadamard gate (Quantum Fourier transform for single qudit).
        This creates uniform superpositions and performs basis transformation.
        
        According to the paper: 
        (H_d)_jk = (1/√d) × ω^{(d-j)×k} for j,k = 0,1,...,d-1
        where the rows are ordered in decreasing powers of ω.
        
        Args:
            dimension: Dimension of the qudit (d)
            
        Returns:
            d × d unitary matrix representing the Hadamard gate
        """
        H = np.zeros((dimension, dimension), dtype=complex)
        omega = QuditGates.get_omega(dimension)
        
        # Fill the matrix according to the paper's formula
        # Note: The paper uses decreasing powers in rows: row j uses ω^(d-j)
        for row in range(dimension):
            for col in range(dimension):
                # The paper's indexing: row j corresponds to power (d-1-j)
                power = (dimension - 1 - row) * col
                H[row, col] = (omega ** power) / np.sqrt(dimension)
                
        return H
    
    @staticmethod
    def validate_unitary(gate: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Validate that a gate matrix is unitary.
        A matrix U is unitary if U†U = I, where U† is the conjugate transpose.
        
        Args:
            gate: Matrix to validate
            tolerance: Numerical tolerance for equality check
            
        Returns:
            True if the matrix is unitary within tolerance
        """
        identity = np.eye(gate.shape[0])
        product = gate.conj().T @ gate
        return np.allclose(product, identity, atol=tolerance)
    

        """
        Demonstrate how these gates act on basis states.
        """
        print("Qudit Gate Application Examples")
        print("=" * 40)
        
        # Test with a qutrit (d=3)
        d = 3
        print(f"Testing with dimension d={d}")
        
        X = QuditGates.x_gate(d)
        Z = QuditGates.z_gate(d)  
        H = QuditGates.hadamard_gate(d)
        
        print(f"X gate (shift operator):")
        print(X)
        print(f"Z gate (phase operator):")
        print(Z)
        print(f"H gate (Fourier transform):")
        print(np.round(H, 3))  # Round for readability
        
        # Validate unitarity
        print(f"X is unitary: {QuditGates.validate_unitary(X)}")
        print(f"Z is unitary: {QuditGates.validate_unitary(Z)}")
        print(f"H is unitary: {QuditGates.validate_unitary(H)}")
        
        # Show phase factor
        omega = QuditGates.get_omega(d)
        print(f"ω for d={d}: {omega} (≈ {np.round(omega, 3)})")
        print(f"ω^{d} = {omega**d} (should be 1)")