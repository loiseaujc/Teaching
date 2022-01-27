using Plots; gr()
using Plots.PlotMeasures
# using Pyplot
using LaTeXStrings
using DynamicalSystems
using ForwardDiff
# using Roots

using LinearAlgebra: norm

logistic_map(x, μ, t) = μ[1] * x * (1 - x)
logistic_map(x, μ) = logistic_map(x, μ, 0)
sine_map(x, μ, t) = μ[1] * sin(π*x)

function dynamics_summary(dynsys, n)

    # -->
    p₁ = time_series(dynsys, n)

    # -->
    p₂ = cobweb(dynsys, n)

    # -->
    p = plot(
        p₁, p₂,
        layout=grid(1, 2, widths=[0.75, 0.25]),
        link=:y,
        size=(6*192, 192),
        left_margin=10px,
        bottom_margin=20px,
    )

    return p
end

function time_series(dynsys, n)

    # --> Run a trajectory.
    x = trajectory(dynsys, n)

    # --> Plot the figure.
    p = plot(
        0:n, x,
        color=:red,
        linewidth=2,
        markershape=:circle,
        legend=:none,
        size=(4*192, 192),
        framestyle=:box,
        xlim=(0, n),
        ylim=(0, 1),
        xlabel=L"\textrm{Iteration } k",
        ylabel=L"x_k",
    )

    return p
end

function cobweb(dynsys, n ; y = collect(0:0.001:1))

    # --> Run a trajectory.
    x = trajectory(dynsys, n)

    # -->
    μ = dynsys.p
    f(x) = dynsys.f(x, μ, 0.0)

    # -->
    p = plot(
        y, f.(y),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color=:red,
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f(x)",
    )

    plot!(
        y, y,
        color=:dimgray,
        linewidth=1,
    )

    plot!(
        [x[1], x[1]], [0, x[2]],
        color=:black,
        linewidth=0.5,
        # arrow=arrow(0.2),
    )

    plot!(
        [x[1], x[2]], [x[2], x[2]],
        color=:black,
        linewidth=0.5,
        # arrow=arrow(0.2),
    )

    for i in 2:n
        plot!(
            [x[i], x[i]], [x[i], x[i+1]],
            color=:black,
            linewidth=0.5,
            # arrow=arrow(0.2),
        )

        plot!(
            [x[i], x[i+1]], [x[i+1], x[i+1]],
            color=:black,
            linewidth=0.5,
            # arrow=arrow(0.2),
        )
    end

    return p
end

function orbit_diagram(dynsys, μₛ ; n=500, Ttr=4000)

    # -->
    output = orbitdiagram(dynsys, 1, 1, μₛ; n=n, Ttr=Ttr)

    # -->
    L = length(μₛ)
    x = Vector{Float64}(undef, n*L)
    y = copy(x)

    for j in 1:L
        x[(1 + (j-1)*n):j*n] .= μₛ[j]
        y[(1 + (j-1)*n):j*n] .= output[j]
    end

    # -->
    p = plot(
        x, y,
        linewidth=0,
        color=:transparent,
        markershape=:circle,
        markersize=0.005,
        markercolor=:red,
        markerstrokecolor=:red,
        markeralpha=0.01,
        xlim=(μₛ[1], μₛ[end]),
        ylim=(0, 1),
        legend=:none,
        framestyle=:box,
        size=(4*192, 192),
        xlabel=L"\mu",
        ylabel=L"x"
    )

    return p
end

function fixed_point_convergence(dynsys_stable, dynsys_superstable ; n=25)

    # --> Stable fixed point.
    μ = dynsys_stable.p[1]
    x₀ = (μ - 1) / μ
    x = trajectory(dynsys_stable, n) .- x₀

    # -->
    p = plot(
        0:n, abs.(x),
        ylim=(1e-14, 1),
        yscale=:log10,
        color=:red,
        linewidth=2,
        size=(256, 192),
        label="Stable",
        xlabel=L"\textrm{Iteration } k",
        ylabel=L"\eta_k",
        framestyle=:box,
        fg_legend=:transparent,
        bg_legend=:transparent,
    )

    # --> Stable fixed point.
    μ = dynsys_superstable.p[1]
    x₀ = (μ - 1) / μ
    x = trajectory(dynsys_superstable, n) .- x₀

    # -->
    p = plot!(
        0:n, abs.(x),
        color=:dimgray,
        linestyle=:dash,
        linewidth=2,
        label="Super stable",
    )

    return p
end

function cycle_2_creation()

    # -->
    μ = 1 + √5

    # -->
    x = collect(0:0.001:1)

    f(x) = logistic_map(x, μ)
    f²(x) = f.(f.(x))

    # -->
    p = plot(
        x, f²(x),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f^2(x)"
    )

    p = plot!(
        x, x,
        color="dimgray",
        linewidth=1,
    )

    # -->
    x₁ = (μ + 1 - √((μ-3) * (μ+1))) / (2μ)
    x₂ = (μ + 1 + √((μ-3) * (μ+1))) / (2μ)
    x₃ = 0
    x₄ = 1 - 1 / μ

    p = plot!(
        [x₁], [x₁],
        markershape=:circle,
        markercolor="black"
    )

    p = plot!(
        [x₂], [x₂],
        markershape=:circle,
        markercolor="black"
    )

    p = plot!(
        [x₃], [x₃],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        [x₄], [x₄],
        markershape=:circle,
        markercolor="white"
    )

    return p
end

function flip_bifurcation(μ)

    # -->
    x = collect(0:0.001:1)

    f(x) = logistic_map(x, μ)
    ∂f(x) = μ .- 2μ * x

    # -->
    p = plot(
        x, f.(x),
        xlim=(0.4, 1),
        ylim=(0.4, 1),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f(x)"
    )

    p = plot!(
        x, x,
        color="dimgray",
        linewidth=1,
    )

    # -->
    if μ >= 3
        x₁ = (μ + 1 - √((μ-3) * (μ+1))) / (2μ)
        x₂ = (μ + 1 + √((μ-3) * (μ+1))) / (2μ)
    end
    x₃ = 0
    x₄ = 1 - 1 / μ

    if μ >= 3
        p = plot!(
            [x₁], [f(x₁)],
            markershape=:circle,
            markercolor="black"
        )

        p = plot!(
            [x₂], [f(x₂)],
            markershape=:circle,
            markercolor="black"
        )

        p = plot!(
            [x₂, x₁], [x₁, x₁],
            color="black",
            linewidth=1,
            # arrow=true,
        )

        p = plot!(
            [x₁, x₁], [x₁, x₂],
            color="black",
            linewidth=1,
            # arrow=true,
        )

        p = plot!(
            [x₁, x₂], [x₂, x₂],
            color="black",
            linewidth=1,
            # arrow=true,
        )

        p = plot!(
            [x₂, x₂], [x₂, x₁],
            color="black",
            linewidth=1,
            # arrow=true,
        )
    end

    p = plot!(
        [x₄], [x₄],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        x, x₄ .+ ∂f(x₄).*(x.-x₄),
        color="dimgray",
        linestyle=:dash
    )

    return p
end

function cycle_4_creation()

    # -->
    μ = 1 + √6 + 0.01

    # -->
    x = collect(0:0.001:1)

    f(x) = logistic_map(x, μ)
    f²(x) = f.(f.(x))
    ∂f²(x) = ForwardDiff.derivative(f², x)


    # -->
    p = plot(
        x, f²(x),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f^2(x)"
    )

    p = plot!(
        x, x,
        color="dimgray",
        linewidth=1,
    )

    # -->
    x₁ = (μ + 1 - √((μ-3) * (μ+1))) / (2μ)
    x₂ = (μ + 1 + √((μ-3) * (μ+1))) / (2μ)
    x₃ = 0
    x₄ = 1 - 1 / μ

    p = plot!(
        [x₁], [x₁],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        [x₂], [x₂],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        [x₃], [x₃],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        [x₄], [x₄],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        x, x₁ .+ ∂f²(x₁).*(x.-x₁),
        color="dimgray",
        linestyle=:dash
    )

    p = plot!(
        x, x₂ .+ ∂f²(x₂).*(x.-x₂),
        color="dimgray",
        linestyle=:dash
    )

    return p
end

function lyapunov_time_series(dynsys_1, dynsys_2, n)

    # --> Run two nearby trajectories.
    x = trajectory(dynsys_1, n)
    y = trajectory(dynsys_2, n)

    # -->
    d = norm.(x-y)

    # --> Plot the figure.
    p₁ = plot(
        0:n, x,
        color=:red,
        linewidth=2,
        markershape=:circle,
        legend=:none,
        size=(4*192, 192),
        framestyle=:box,
        xlim=(0, n),
        ylim=(0, 1),
        # xlabel=L"\textrm{Iteration } k",
        ylabel=L"x_k",
        alpha=0.5,
    )

    p₁ = plot!(
        0:n, y,
        color=:dimgray,
        linewidth=2,
        linestyle=:dash,
        markershape=:circle,
        alpha=0.5,
    )

    # -->
    p₂ = plot(
        0:n, d,
        xlim=(0, n),
        ylim=(1e-12, 1),
        yscale=:log10,
        color=:red,
        linewidth=2,
        framestyle=:box,
        legend=:none,
        ylabel=L"\vert x_k -y_k \vert",
        xlabel=L"\textrm{Iteration } k",
    )

    # -->
    p = plot(
        p₁, p₂,
        layout=(2, 1),
        link=:x,
        size=(4*192, 2*192),
        bottom_margin=10px,
    )

    return p
end

function lyapunov_exponent()

     # -->
     dynsys = Systems.logistic()
     μₛ = 0:0.0001:4
     λₛ = zeros(length(μₛ))

     for (i, μ) in enumerate(μₛ)
         set_parameter!(dynsys, 1, μ)
         λₛ[i] = lyapunov(dynsys, 20000 ; Ttr=2000)
     end

     # -->
     p = plot(
        μₛ, λₛ,
        xlim=(0, 4),
        ylim=(-8, 2),
        color=:red,
        linewidth=1,
        size=(4*192, 192),
        framestyle=:box,
        legend=:none,
        xlabel=L"\mu",
        ylabel=L"\lambda",
        bottom_margin=15px,
     )

     p = hline!(
        [0],
        color=:dimgray,
        )

    return p
end

function rossler_bifurcation_diagram()

    dynsys = Systems.roessler(a=0.2, b=0.2, c=5.7)
    pvalues = range(0.001, stop=2, length=1001)
    i = 2
    plane = (1, 0)
    tf = 4000.0
    p_index = 2

    output = produce_orbitdiagram(dynsys, plane, i, p_index, pvalues; tfinal=tf, Ttr=2000.0, printparams=:true)

    p = plot(
        xlim=(pvalues[1], pvalues[end]),
        legend=:none,
        framestyle=:box,
    )

    for (j, p) in enumerate(pvalues)
        plot!(
            fill(p, length(output[j])), output[j],
            linewidth=0,
            color=:transparent,
            markershape=:circle,
            markersize=0.005,
            markercolor=:red,
            markerstrokecolor=:red,
            markeralpha=0.025,
        )
    end

    plot!(
        size=(4*192, 2*192),
        guidefont=font(20),
        tickfont=font(16),
        xlabel=L"b",
        ylabel=L"y",
        ylim=(-15, 0),
        bottom_margin=10px,
        right_margin=10px,
    )

    return p
end

function type_I_intermittency()

    f(x, ϵ, t) = ϵ + x + x^2
    dynsys = DiscreteDynamicalSystem(f, -0.5, 0.01)
    p = cobweb(dynsys, 100 ; y = collect(-1:0.001:1))
    plot!(xlim=(-1, 1))
    plot!(ylim=(-1, 1))

    return p
end

function type_II_intermittency()

    f(x, ϵ, t) = (1 + ϵ) * x + x^3
    dynsys = DiscreteDynamicalSystem(f, 0.01, 0.1)
    p = cobweb(dynsys, 100 ; y = collect(-1:0.001:1))
    plot!(xlim=(-1, 1))
    plot!(ylim=(-1, 1))

    return p
end


function type_III_intermittency()

    f(x, ϵ, t) = -(1 + ϵ) * x - x^3
    dynsys = DiscreteDynamicalSystem(f, 0.2, 0.001)
    p = cobweb(dynsys, 100 ; y = collect(-1:0.001:1))
    plot!(xlim=(-1, 1))
    plot!(ylim=(-1, 1))

    return p
end

function renormalization_1_cycle()

    # -->
    x = collect(0:0.001:1)
    f(x) = logistic_map(x, 2)

    # -->
    p = plot(
        x, f.(x),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color=:red,
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f(x)",
        ticks=:false,
    )

    plot!(
        x, x,
        color=:dimgray,
        linewidth=1,
    )

    plot!(
        [0.5], [0.5],
        markershape=:circle,
        markercolor=:black,
    )

    return p
end

function renormalization_2_cycle()
    p = cycle_2_creation()
    plot!(ticks=:false)

    μ = 1 + √5
    x₀ = (μ - 1) / μ
    Δ = x₀ - 0.5

    rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])
    plot!(
        rectangle(2Δ, 2Δ, x₀-2Δ, x₀-2Δ),
        color=:transparent,
    )

    return p
end

function renormalization_2_cycle_bis()

    μ = 1 + √5
    x₀ = (μ - 1) / μ
    Δ = x₀ - 0.5

    p = cycle_2_creation()
    plot!(
        xlim=(0.5 - Δ, 0.5 + Δ),
        ylim=(0.5-Δ, x₀),
        yflip=:true,
        xflip=:true,
        ticks=:false,
        xlabel=L"x / \alpha",
        ylabel=L"\alpha f^2(x / \alpha)",
    )

    return p
end

function renormalization_normal_form()

    f(x, ϵ, t) = -(1 + ϵ) * x + x^2
    dynsys = DiscreteDynamicalSystem(f, 0.0, 0.0)
    p = cobweb(dynsys, 1 ; y=collect(-1:0.001:2))
    plot!(
        xlim=(-1, 2),
        ylim=(-1, 2),
    )

    plot!(
        [0], [0],
        markershape=:circle,
        markercolor=:white,
    )

    return p
end

function renormalization_exercise_sequence()

    f(ϵ, p, t) = -2.0 + √(6.0 + ϵ)
    dynsys = DiscreteDynamicalSystem(f, 0.0, 0.0)
    p = cobweb(dynsys, 100)
    plot!(
        xlabel=L"\epsilon",
        ylabel=L"f(\epsilon)"
    )

    return p
end

function renormalization_flip_bifurcation(ϵ)

    # -->
    x = collect(-1:0.001:2)

    f(x) = -(1 + ϵ) * x + x^2
    ∂f(x) = -(1 + ϵ) - 2*x

    # -->
    p = plot(
        x, f.(x),
        xlim=(-1, 2),
        ylim=(-1, 2),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f(x)"
    )

    p = plot!(
        x, x,
        color="dimgray",
        linewidth=1,
    )

    # -->
    if ϵ >= 0
        x₁ = (ϵ + √(ϵ^2 + 4ϵ)) / 2
        x₂ = (ϵ - √(ϵ^2 + 4ϵ)) / 2
    end
    x₃ = 0

    if ϵ >= 0
        p = plot!(
            [x₁], [f(x₁)],
            markershape=:circle,
            markercolor="black"
        )

        p = plot!(
            [x₂], [f(x₂)],
            markershape=:circle,
            markercolor="black"
        )

        p = plot!(
            [x₂, x₁], [x₁, x₁],
            color="black",
            linewidth=1,
            # arrow=true,
        )

        p = plot!(
            [x₁, x₁], [x₁, x₂],
            color="black",
            linewidth=1,
            # arrow=true,
        )

        p = plot!(
            [x₁, x₂], [x₂, x₂],
            color="black",
            linewidth=1,
            # arrow=true,
        )

        p = plot!(
            [x₂, x₂], [x₂, x₁],
            color="black",
            linewidth=1,
            # arrow=true,
        )
    end

    p = plot!(
        [x₃], [x₃],
        markershape=:circle,
        markercolor="white"
    )

    p = plot!(
        x, x₃ .+ ∂f(x₃).*(x.-x₃),
        color="dimgray",
        linestyle=:dash
    )

    plot!(
        xlim=(-0.2, 0.2),
        ylim=(-0.2, 0.2),
    )

    return p
end

function renormalization_2_cycle_ter()
    # -->
    x = collect(-1:0.001:2)

    ϵ = -2 + √6

    f(x) = -(1 + ϵ) * x + x^2
    f²(x) = f(f(x))
    ∂f²(x) = ForwardDiff.derivative(f², x)
    ∂²f²(x) = ForwardDiff.derivative(∂f², x)
    # -->
    p = plot(
        x, f².(x),
        xlim=(-1, 2),
        ylim=(-1, 2),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f^2(x)"
    )

    p = plot!(
        x, x,
        color="dimgray",
        linewidth=1,
    )

    x₁ = (ϵ + √(ϵ^2 + 4ϵ)) / 2
    x₂ = (ϵ - √(ϵ^2 + 4ϵ)) / 2
    x₃ = 0

    p = plot!(
        [x₁], [f²(x₁)],
        markershape=:circle,
        markercolor=:white
    )

    p = plot!(
        [x₂], [f²(x₂)],
        markershape=:circle,
        markercolor=:white
    )

    p = plot!(
        [0], [0],
        markershape=:circle,
        markercolor=:white
    )

    p = plot!(
        x, f²(x₁) .+ ∂f²(x₁).*(x.-x₁) + 0.5.*∂²f²(x₁).*(x.-x₁).^2,
        color=:blue,
        alpha=0.5,
        linewidth=1,
        linestyle=:dash
    )

    return p
end

function functional_equation()

    a₁ = (-1 - √3) / 2
    α = 2a₁

    g(x) = 1 + a₁*x^2

    x = collect(-1:0.001:1)

    p = plot(
        x, g.(x),
        xlim=(-1, 1),
        ylim=(-0.5, 1.5),
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"g(x)",
        label=L"g(x)",
        fg_legend=:transparent,
        bg_legend=:transparent,
        legend=:bottomright,
    )

    p = plot!(
        x, α * g.(g.(x./α)),
        color=:blue,
        linestyle=:dash,
        alpha=0.5,
        label=L"\alpha g^2(x/\alpha)",
    )

    return p
end

function main()

    #########################################
    #####                               #####
    #####     ILLUSTRATING DYNAMICS     #####
    #####                               #####
    #########################################

    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, 0.9)
    # dynamics_summary(dynsys, 100)
    # savefig("../Slides/imgs/dynamics_a.pdf")
    #
    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.1, 2.5)
    # dynamics_summary(dynsys, 100)
    # savefig("../Slides/imgs/dynamics_b.pdf")
    #
    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, 3.25)
    # dynamics_summary(dynsys, 100)
    # savefig("../Slides/imgs/dynamics_c.pdf")
    #
    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, 3.5)
    # dynamics_summary(dynsys, 100)
    # savefig("../Slides/imgs/dynamics_d.pdf")
    #
    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, 3.9)
    # dynamics_summary(dynsys, 100)
    # savefig("../Slides/imgs/dynamics_e.pdf")

    #################################
    #####                       #####
    #####     ORBIT DIAGRAM     #####
    #####                       #####
    #################################

    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, [3.9])
    # μₛ = 0:0.0001:4
    # p = orbit_diagram(dynsys, μₛ ; n=500)
    #
    # # -->
    # plot!(
    #     size=(4*192, 2*192),
    #     guidefont=font(20),
    #     tickfont=font(16),
    # )
    # savefig("../Slides/imgs/orbit_diagram_a.png")

    #######################################################
    #####                                             #####
    #####     STABLE VS SUPER-STABLE FIXED POINTS     #####
    #####                                             #####
    #######################################################

    # # --> Stable fixed point.
    # μ = 1.5
    # x₀ = (μ - 1) / μ + 0.1
    # dynsys_stable = DiscreteDynamicalSystem(logistic_map, x₀, μ)
    #
    # # --> Super stable fixed point.
    # μ = 2.0
    # x₀ = (μ - 1) / μ + 0.1
    # dynsys_superstable = DiscreteDynamicalSystem(logistic_map, x₀, μ)
    #
    # fixed_point_convergence(dynsys_stable, dynsys_superstable)
    # savefig("../Slides/imgs/stable_vs_superstable.pdf")

    ####################################
    #####                          #####
    #####     2-CYCLE CREATION     #####
    #####                          #####
    ####################################

    # # -->
    # cycle_2_creation()
    # savefig("../Slides/imgs/cycle_2_creation.pdf")
    #
    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, 1 + √5)
    # time_series(dynsys, 1000)
    # plot!(xlim=(900, 1000))
    # savefig("../Slides/imgs/cycle_2_time_series.pdf")
    #
    # # -->
    # flip_bifurcation(3.05)
    # savefig("../Slides/imgs/flip_bifurcation.pdf")
    #
    # # -->
    # cycle_4_creation()
    # savefig("../Slides/imgs/cycle_4_creation.pdf")
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, 1 + √6 + 0.01)
    # time_series(dynsys, 1000)
    # plot!(xlim=(900, 1000))
    # savefig("../Slides/imgs/cycle_4_time_series.pdf")

    ###################################
    #####                         #####
    #####     RENORMALIZATION     #####
    #####                         #####
    ###################################

    renormalization_1_cycle()
    savefig("../Slides/imgs/renormalization_1.pdf")

    renormalization_2_cycle()
    savefig("../Slides/imgs/renormalization_2.pdf")

    renormalization_2_cycle_bis()
    savefig("../Slides/imgs/renormalization_3.pdf")

    # -->
    functional_equation()
    savefig("../Slides/imgs/quadratic_functional_equation.pdf")

    # -->
    renormalization_normal_form()
    savefig("../Slides/imgs/renormalization_normal_form.pdf")

    # -->
    renormalization_flip_bifurcation(0.01)
    savefig("../Slides/imgs/renormalization_flip_bifurcation.pdf")

    # -->
    renormalization_2_cycle_ter()
    savefig("../Slides/imgs/renormalization_2_cycle.pdf")

    # -->
    renormalization_exercise_sequence()
    savefig("../Slides/imgs/renormalization_epsilon_sequence.pdf")

    ######################################
    #####                            #####
    #####     LYAPUNOV EXPONENTS     #####
    #####                            #####
    ######################################

     # -->
     dynsys = DiscreteDynamicalSystem(logistic_map, 0.9, 3.7)
     cobweb(dynsys, 100)
     savefig("../Slides/imgs/into_the_chaos.pdf")

    # # -->
    # dynsys_1 = DiscreteDynamicalSystem(logistic_map, 0.1, 3.7)
    # dynsys_2 = DiscreteDynamicalSystem(logistic_map, 0.1 + 1e-12, 3.7)
    # lyapunov_time_series(dynsys_1, dynsys_2, 200)
    # savefig("../Slides/imgs/lyapunov_time_series.pdf")
    #
    # # -->
    # lyapunov_exponent()
    # savefig("../Slides/imgs/lyapunov_exponents.pdf")


    ####################################
    #####                          #####
    #####     3-CYCLE CREATION     #####
    #####                          #####
    ####################################

    # # -->
    # plot!(
    #     xlim=(3.5, 4),
    #     ylim=(-2, 2),
    #     size=(192, 192),
    # )
    # savefig("../Slides/imgs/lyapunov_exponents_zoom.pdf")
    #
    # # -->
    # dynsys = DiscreteDynamicalSystem(logistic_map, 0.5, [3.9])
    # μₛ = 3.8:0.0001:3.9
    # p = orbit_diagram(dynsys, μₛ ; n=1000)
    #
    # # -->
    # plot!(
    #     size=(2*192, 2*192),
    #     guidefont=font(20),
    #     tickfont=font(16),
    #     xticks=3.8:0.1:3.9,
    #     yticks=0:0.5:1,
    #     # bottom_margin=1px,
    #     # left_margin=1px,
    #     # right_margin=1px,
    #     # top_margin=1px,
    # )
    # savefig("../Slides/imgs/orbit_diagram_b.png")


    #################################
    #####                       #####
    #####     INTERMITTENCY     #####
    #####                       #####
    #################################

    ################################
    #####                      #####
    #####     UNIVERSALITY     #####
    #####                      #####
    ################################

    # # -->
    # dynsys = DiscreteDynamicalSystem(sine_map, 0.5, [0.9])
    # μₛ = 0:0.0001:1
    # p = orbit_diagram(dynsys, μₛ ; n=500)
    #
    # # -->
    # plot!(
    #     size=(4*192, 2*192),
    #     guidefont=font(20),
    #     tickfont=font(16),
    # )
    # savefig("../Slides/imgs/sine_map_orbit_diagram.png")
    #
    # # -->
    # rossler_bifurcation_diagram()
    # savefig("../Slides/imgs/rossler_orbit_diagram.png")

     return
end
