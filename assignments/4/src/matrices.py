import sympy as sp
import numpy as np

def get_fi_Q(f_in, l_in):
    """
    Get state transition matrix fi and dynamic model covariance matrix Q.
    Author: Jernej Vivod

    Args:
        f_in (numpy.ndarray): The F matrix (cont. transition matrix)
        l_in (numpy.ndarray): The L matrix (noise specification matrix)

    Returns:
        (tuple): State transition matrix fi and dynamic model 
        covariance matrix Q.
    """

    # Parse matrices.
    f = sp.Matrix(f_in)
    l = sp.Matrix(l_in)

    # Define symbols.
    t, q = sp.symbols('t q')

    # Compute fi using matrix exponentiation.
    fi = sp.exp(f*t)

    # Compute Q.
    Q = sp.integrate((fi*l)*q*(fi*l).T, (t, 0, t))

    # Return computed results.
    return fi.subs({t:1}), Q.subs({t:1})


