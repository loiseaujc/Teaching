# -->
using DifferentialEquations

# -->
using LinearAlgebra

# -->
using Plots; pyplot()
using LaTeXStrings

function pendulum_dynamics!(du, u, p, t)

    # --> Unpack parameters.
    k, ω₀ = p

    # --> Equations of motion.
    du[1] = u[2]
    du[2] = -2k * u[2] - ω₀^2 * sin.(u[1])

    return du
end

function main()

    # -->
    tspan = (0.0, 25.0)

    # --> Parameters.
    k, ω₀ = 0.25, 1.0
    params = (k, ω₀)

    # -->
    p = plot()

    for i ∈ -3:3
        for j ∈ -3:3
            x₀ = [i * π/4, j * π/4]

            # --> Define the problem.
            pendulum_prob = ODEProblem(pendulum_dynamics!, x₀, tspan, p=params)

            # --> Simulated the problem.
            pendulum_sol = solve(pendulum_prob, Tsit5())

            p = plot!(
                pendulum_sol, vars=(1, 2),
                color=:chartreuse,
                legend=false
            )
        end
    end

    return p
end
