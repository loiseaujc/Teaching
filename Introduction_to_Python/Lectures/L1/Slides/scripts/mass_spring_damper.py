import numpy as np
from scipy.integrate import solve_ivp

def mass_spring_damper(t, u, k):
    # --> Unpack variables.
    x, dx = u
    # --> Equations of motion.
    ddx = -x - 2*k*dx
    return dx, ddx

# --> Initial conditions.
u = np.array([-1.0, 0.0])

# --> Parameters.
T = (0.0, 100)
k = 0.05

# --> Simulate the system.
output = solve_ivp(
    lambda t, u : dynsys(t, u, k),
    T,
    u
)
