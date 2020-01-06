# -->
using DifferentialEquations

# -->
using LinearAlgebra

# -->
using Plots; pyplot()
using LaTeXStrings

function van_der_pol!(du, u, μ, t)

    # -->
    du[1] = u[2]
    du[2] = μ*(1 - u[1].^2).*u[2] - u[1]

    return du
end

function main()

    # -->
    tspan = (0.0, 1000.0)
    x₀ = [0.1, 0.1]

    μ_range = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    p = plot(size=(600, 1200))

    for i = 1:length(μ_range)
        μ = μ_range[i]
        α = i / length(μ_range)

        prob = ODEProblem(van_der_pol!, x₀, tspan, p=μ)
        sol = solve(prob, Tsit5())

        p = plot!(
            sol, vars=(1, 2),
            tspan=(950, 1000),
            color=:black,
            aspect_ratio=:equal,
            linewidth=2,
            alpha=α,
            tickfont=font(28),
            guidefont=font(32),
            colorbar=false,
        )
    end

    p = plot!(
        xlabel=L"x",
        ylabel=L"\dot{x}",
        legend=:none,
        framestyle=:box
    )

    png("../Slides/imgs/van_der_pol_limit_cycle")
    return p
end
