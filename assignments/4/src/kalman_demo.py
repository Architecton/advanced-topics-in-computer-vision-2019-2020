import numpy as np
from ex4_utils import kalman_step
from matrices import get_fi_Q
import argparse

def add_noise(y, sig):
    """
    Add gaussian noise to vector of values.

    Args:
        y (numpy.ndarray): Vector of values to which to add Gaussian noise.
        x (numpy.ndarray):
    """
    # return y + np.random.normal(size=len(y), scale=sig)
    y[::len(y)//120] += np.random.normal(size=int(round(len(y)/(len(y)//120))), scale=sig)
    return y

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=500)
parser.add_argument('--model', type=str, default='ncv')
parser.add_argument('--example', type=str, default='noisy_line')
parser.add_argument('--xlim', nargs=2, type=float)
parser.add_argument('--ylim', nargs=2, type=float)
parser.add_argument('--sanity-check', action='store_true')
args = parser.parse_args()


# Select sample data.
if args.example == 'noisy_line':
    x = np.linspace(0, 5, args.N)
    y_true = x.copy()
    y = add_noise(y_true, 0.3)

elif args.example == 'spiral':
    v = np.linspace(5 * np.pi, 0, args.N)
    x = np.cos(v) * v
    y = y_true = np.sin(v) * v


# Set matrices F and L for specified model.
if args.model == 'rw':
    # RANDOM WALK MODEL (NEAR CONSTANT POSITION)
    f = np.zeros((2, 2))
    l = np.eye(2)
    
    # Set measurement length.
    MEAS_LEN = 2
    
    # Allocate arrays for storing results.
    sx = np.zeros(len(x), dtype=np.float32)
    sy = np.zeros(len(y), dtype=np.float32)

    # Set initial solution values (known starting position).
    sx[0] = x[0]
    sy[0] = y[0]

elif args.model == 'ncv':
    # NEAR CONSTANT VELOCITY MODEL
    f = np.array([[0, 0, 1, 0], 
                  [0, 0, 0, 1], 
                  [0, 0, 0, 0], 
                  [0, 0, 0, 0]])
    l = np.array([[0, 0], 
                  [0, 0], 
                  [1, 0], 
                  [0, 1]])
    
    # Set measurement length. 
    MEAS_LEN = 4
    
    # Allocate arrays for storing results.
    sx = np.zeros(len(x), dtype=np.float32)
    sy = np.zeros(len(y), dtype=np.float32)
    sx_d = np.zeros(len(x), dtype=np.float32)
    sy_d = np.zeros(len(y), dtype=np.float32)

    # Set initial solution values (known starting position and velocity).
    sx[0] = x[0]
    sy[0] = y[0]
    sx_d[0] = x[1] - x[0]
    sy_d[0] = y[1] - y[0]

elif args.model == 'nca':
    # NEAR CONSTANT ACCELERATION MODEL
    f = np.array([[0, 0, 1, 0, 0, 0], 
                  [0, 0, 0, 1, 0, 0], 
                  [0, 0, 0, 0, 1, 0], 
                  [0, 0, 0, 0, 0, 1], 
                  [0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0]])
    l = np.array([[0, 0], 
                  [0, 0], 
                  [0, 0], 
                  [0, 0], 
                  [1, 0], 
                  [0, 1]])
    
    # Set measurement length.
    MEAS_LEN = 6

    # Allocate arrays for storing results.
    sx = np.zeros(len(x), dtype=np.float32)
    sy = np.zeros(len(y), dtype=np.float32)
    sx_d = np.zeros(len(x), dtype=np.float32)
    sy_d = np.zeros(len(y), dtype=np.float32)
    sx_dd = np.zeros(len(x), dtype=np.float32)
    sy_dd = np.zeros(len(y), dtype=np.float32)

    # Set initial solution values (known starting position, velocity and acceleration).
    sx[0] = x[0]
    sy[0] = y[0]
    sx_d[0] = x[1] - x[0]
    sy_d[0] = y[1] - y[0]
    sx_dd[0] = (x[2] - x[1]) - (x[1] - x[0])
    sy_dd[0] = (y[2] - y[1]) - (y[1] - y[0])


# Compute required matrices.
A_sym, Q_sym = get_fi_Q(f, l)  # Get transition matrix and system covariance matrix.
q = 3/4*0.5**2
Q_i = np.array(Q_sym.subs({'q':q})).astype(float)
A = np.array(A_sym).astype(float)
C = np.hstack((np.eye(MEAS_LEN), np.zeros((MEAS_LEN, A.shape[0]-MEAS_LEN), dtype=float)))
R_i = 20.84*np.eye(MEAS_LEN)  # Get observation model covariance matrix.

# Set initial state.
state = np.zeros((A.shape[0], 1), dtype=np.float32)
state[0] = x[0]
state[1] = y[0]
if args.model in {'ncv', 'nca'}:
    state[2] = 0 # x[1] - x[0]
    state[3] = 0 # y[1] - y[0]
    if args.example == 'nca':
        state[4] = 0 # (x[1] - x[0]) - (x[2] - x[1])
        state[5] = 0 # (y[1] - y[0]) - (y[2] - y[1])

# Set initial posterior covariance.
covariance = np.eye(A.shape[0], dtype=np.float32)

# If doing sanity chack with known working implementation.
if args.sanity_check:
    from filterpy.kalman import predict, update

for j in range(1, x.size):

    ### PERFORM MEASUREMENT ###
    
    # If random walk model, measure position.
    if args.model == 'rw':
        meas = np.array([[x[j]], 
                         [y[j]]])

    # If near constant velocity model, measure position and velocity.
    elif args.model == 'ncv':
        meas = np.array([[x[j]], 
                         [y[j]], 
                         [x[j] - x[j-1]], 
                         [y[j] - y[j-1]]])

    # If near constant acceleration model, measure position, velocity and acceleration.
    elif args.model == 'nca':
        meas = np.array([[x[j]], 
                         [y[j]], 
                         [x[j]-x[j-1]], 
                         [y[j]-y[j-1]], 
                         [(x[j] - x[j-1]) - (x[j-1] - x[j-2])], 
                         [(y[j] - y[j-1]) - (y[j-1] - y[j-2])]])
    
    # If doing sanity check, use library. Else use provided kalman_step function.
    if args.sanity_check:
        state, covariance = predict(state, covariance, A, Q_i)
        state, covariance = update(state, covariance, meas, R_i, C)
    else:
        state, covariance, _ , _ = kalman_step(A, C, Q_i, R_i, meas, state, covariance)

    
    # Get position (velocity, acceleration) from deduced state.
    sx[j] = state[0]
    sy[j] = state[1]
    if args.model in {'ncv', 'nca'}:
        sx_d[j] = state[2]
        sy_d[j] = state[3]
        if args.model == 'nca':
            sx_dd[j] = state[4]
            sy_dd[j] = state[5]


### Plot results. ###
import matplotlib.pyplot as plt

# Plot measured positions and estimated positions.
plt.figure(0)
plt.plot(x, y, '.', label='original')
plt.plot(sx, sy, '.', label='filtered')
if args.xlim:
    plt.xlim(args.xlim)
if args.ylim:
    plt.ylim(args.ylim)
plt.legend()

# Plot estimated velocity (and acceleration if nca model).
if args.model in {'ncv', 'nca'}:
    plt.figure(1)
    plt.plot(sx_d, label='Velocity in x')
    plt.plot(sy_d, label='Velocity in y')
    plt.xlabel('time step')
    plt.ylabel('Estimated velocity')
    plt.legend()
    if args.model == 'nca':
        plt.figure(2)
        plt.plot(sx_dd, label='Acceleration in x')
        plt.plot(sy_dd, label='Acceleration in y')
        plt.xlabel('time step')
        plt.ylabel('Estimated acceleration')
        plt.legend()
plt.show()

