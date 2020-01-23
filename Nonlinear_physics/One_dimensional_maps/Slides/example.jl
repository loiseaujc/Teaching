using DynamicalSystems

# Define f(x, r, t).
f(x, r, t) = r*x*(1-x)

# Initial condition and parameter.
x, r = 0.1, 3.5

# Create dynamical system.
dynsys = DiscreteDynamicalSystem(f, x, r)

# Simulate system.
trajectory(dynsys, 100)
