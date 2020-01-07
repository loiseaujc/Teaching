# -->
using DifferentialEquations

# -->
using LinearAlgebra

# -->
using Plots; gr()
using LaTeXStrings

function dynamics!(du, u, p, t)

    # --> Unpack parameters.
    k, ω₀ = p

    # --> Equations of motion.
    du[1] = u[2]
    du[2] = -2k * u[2] - ω₀^2 * sin.(u[1])

    return du
end

function polar_to_cartesian(t, sol)

    x = sin.(sol(t)[1, :])
    y = -cos.(sol(t)[1, :])

    return [x y]
end

function plot_pendulum(i, x)

    p = plot(
        [0, x[i, 1]], [0, x[i, 2]],
        size=(512, 512),
        xlim=(-1.1, 1.1),
        ylim=(-1.5, 1.5),
        color=1,
        label="",
        # markersize=15,
        # markershape=:circle,
        # label="",
        # framestyle=:none,
        # aspect_ratio=:equal
    )

    p = plot!(
        [x[i, 1], x[i, 1]], [x[i, 2], x[i, 2]],
        size=(512, 512),
        xlim=(-1.175, 1.175),
        ylim=(-1.5, 1.5),
        color=1,
        markersize=15,
        markershape=:circle,
        label="",
        framestyle=:none,
        # aspect_ratio=:equal
    )

    # -->
    if i > 9*2
        p = plot!([x[i-3*2:i, 1]], [x[i-3*2:i, 2]],alpha = 0.15,linewidth = 2, color = :red,label ="");
        p = plot!([x[i-5*2:i-3*2, 1]], [x[i-5*2:i-3*2, 2]],alpha = 0.08,linewidth = 2, color = :red,label ="");
        p = plot!([x[i-7*2:i-5*2, 1]], [x[i-7*2:i-5*2, 2]],alpha = 0.04,linewidth = 2, color = :red, label ="");
        p = plot!([x[i-9*2:i-7*2, 1]], [x[i-9*2:i-7*2, 2]],alpha = 0.01,linewidth = 2, color = :red, label="");
    end

    return
end

function main()

    # -->
    tspan = (0.0, 25.0)
    Δt = 0.025
    t = collect(tspan[1]:Δt:tspan[2])

    # --> Parameters.
    k, ω₀ = 0.01, 1.0
    params = (k, ω₀)

    # -->
    x₀ = [3π/4, 0.0]

    # --> Define the problem.
    prob = ODEProblem(dynamics!, x₀, tspan, p=params)

    # --> Simulated the problem.
    sol = solve(prob, Tsit5())

    # -->
    x = polar_to_cartesian(t, sol)

    # -->
    # anim = @animate for i = 1:length(t)
    #     p = plot_pendulum(i, x);
    #
    #     if i == 250
    #         savefig("pendulum_image.png")
    #     end
    # end
    #
    # # -->
    # gif(anim, "single_pendulum.gif", fps=60)
    # mp4(anim, "single_pendulum.mp4", fps=60)

    p = plot(
        [-0.05, -0.05], [0, -1],
        color=1,
        label="",
    )

    p = plot!(
        [-0.05, -0.05], [-1, -1],
        color=1,
        markersize=15,
        markershape=:circle,
        label="",
    )

    p = plot!(
        [0.05, 0.05], [0, 1],
        color=2,
        label="",
    )

    p = plot!(
        [0.05, 0.05], [1, 1],
        size=(128, 512),
        xlim=(-0.15, 0.15),
        ylim=(-1.175, 1.175),
        color=2,
        markersize=15,
        markershape=:circle,
        label="",
        framestyle=:none,
        aspect_ratio=:equal
    )

    savefig("pendulum_equilibria.png")

    return
end
