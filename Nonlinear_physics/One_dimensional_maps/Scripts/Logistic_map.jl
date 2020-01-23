using Plots; gr()
using Plots.PlotMeasures
using LaTeXStrings
using ForwardDiff
using Roots

function logistic_map(x, μ)
    return μ * x .* ( 1 .- x )
end

function cycle_2_creation()

    # -->
    μ = 3.25

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
        x, f(x),
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
            arrow=true,
        )

        p = plot!(
            [x₁, x₁], [x₁, x₂],
            color="black",
            linewidth=1,
            arrow=true,
        )

        p = plot!(
            [x₁, x₂], [x₂, x₂],
            color="black",
            linewidth=1,
            arrow=true,
        )

        p = plot!(
            [x₂, x₂], [x₂, x₁],
            color="black",
            linewidth=1,
            arrow=true,
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

function cycle_3_creation()

    # -->
    μ = 3.835

    # -->
    x = collect(0:0.001:1)

    f(x) = logistic_map(x, μ)
    f³(x) = f.(f.(f.(x)))

    F³(x) = x - f³(x)

    # -->
    p = plot(
        x, f³(x),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f^3(x)"
    )

    p = plot!(
        x, x,
        color="dimgray",
        linewidth=1,
    )

    # --> Spurious fixed points.
    x₁, x₂ = 0, 1 - 1 / μ

    p = plot!(
        [x₁], [x₁],
        markershape=:square,
        markercolor="white",
        markersize=2,
    )

    p = plot!(
        [x₂], [x₂],
        markershape=:square,
        markercolor="white",
        markersize=2,
    )

    # --> Find fixed points.
    x₁ = find_fixed_point(F³, 0.1)
    x₂ = find_fixed_point(F³, 0.2)
    x₃ = find_fixed_point(F³, 0.4)
    x₄ = find_fixed_point(F³, 0.6)
    x₅ = find_fixed_point(F³, 0.9)
    x₆ = find_fixed_point(F³, 0.975)

    p = plot!(
        [x₁], [x₁],
        markershape=:circle,
        markercolor="black",
        markersize=3,
    )

    p = plot!(
        [x₂], [x₂],
        markershape=:circle,
        markercolor="white",
        markersize=3,
    )

    p = plot!(
        [x₃], [x₃],
        markershape=:circle,
        markercolor="black",
        markersize=3,
    )

    p = plot!(
        [x₄], [x₄],
        markershape=:circle,
        markercolor="white",
        markersize=3,
    )

    p = plot!(
    [x₆], [x₆],
    markershape=:circle,
    markercolor="black",
    markersize=3,
    )

    p = plot!(
        [x₅], [x₅],
        markershape=:circle,
        markercolor="white",
        markersize=3,
    )


    return p
end

function find_fixed_point(f, x₀)

    # -->
    D(f) = x -> ForwardDiff.derivative(f, float(x))

    # -->
    x = find_zero((f, D(f)), x₀, Roots.Newton())

    return x
end

function dynamics(μ, x₀, n)

    # -->
    p₁ = time_series(μ, x₀, n)
    p₁ = plot!(
        xlim=(0, n),
        )

    # -->
    f(x) = logistic_map(x, μ)
    p₂ = cobweb(f, x₀, n; plot_arrow=false)
    p₃ = plot!(ylabel=L"f(x)")

    p = plot(
        p₁, p₂,
        layout = grid(1, 2, widths=[0.7, 0.3]),
        size=(6*192, 192),
        link=:y,
        bottom_margin=20px,
        )
    savefig("test.pdf")

    return p
end

function time_series(μ, x₀, n)

    # -->
    f(x) = logistic_map(x, μ)

    # -->
    x = zeros(n)
    x[1] = x₀
    for i in 2:n
        x[i] = f(x[i-1])
    end

    # -->
    p = plot(
        x,
        color=:red,
        linewidth=2,
        markershape=:circle,
        legend=:none,
        size=(4*192, 192),
        framestyle=:box,
        ylim=(0, 1),
        xlabel=L"\textrm{Iteration } k",
        ylabel=L"x_k",
    )

    return p
end

function cobweb(f, x₀, n; plot_arrow=:true)

    # -->
    x = zeros(n)
    x[1] = x₀
    for i in 2:n
        x[i] = f(x[i-1])
    end

    # -->
    y = collect(0:0.001:1)

    # -->
    p = plot(
        y, f.(y),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(192, 192),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f^3(x)"
        )

    p = plot!(
        y, y,
        color="dimgray",
        linewidth=2,
    )

    i = 0
    p = plot!(
        [x[i+1], x[i+1]], [0, f(x[i+1])],
        color="black",
        linewidth=1,
        arrow=plot_arrow,
        )
    p = plot!(
        [x[i+1], f(x[i+1])], [f(x[i+1]), f(x[i+1])],
        color="black",
        linewidth=1,
        arrow=plot_arrow,
        )

    for i in 1:(n-1)
        p = plot!(
            [x[i+1], x[i+1]], [x[i+1], f(x[i+1])],
            color="black",
            linewidth=1,
            arrow=plot_arrow,
            )
        p = plot!(
            [x[i+1], f(x[i+1])], [f(x[i+1]), f(x[i+1])],
            color="black",
            linewidth=1,
            arrow=plot_arrow,
            )
    end

    return p
end

function main()

    # -->
    dynamics(0.9, 0.5, 100)
    savefig("../Slides/imgs/dynamics_a.pdf")

    # -->
    dynamics(2.5, 0.1, 100)
    savefig("../Slides/imgs/dynamics_b.pdf")

    # -->
    dynamics(3.25, 0.1, 100)
    savefig("../Slides/imgs/dynamics_c.pdf")

    # -->
    dynamics(3.5, 0.1, 100)
    savefig("../Slides/imgs/dynamics_d.pdf")

    # -->
    dynamics(3.9, 0.1, 100)
    savefig("../Slides/imgs/dynamics_e.pdf")

    # -->
    cycle_2_creation()
    savefig("../Slides/imgs/cycle_2_creation.pdf")

    # -->
    p = time_series(3.25, 0.5, 1000)
    p = plot!(xlim=(900, 1000))
    savefig("../Slides/imgs/cycle_2_time_series.pdf")

    # -->
    flip_bifurcation(3.05)
    savefig("../Slides/imgs/flip_bifurcation.pdf")

    # -->
    cycle_3_creation()
    savefig("../Slides/imgs/cycle_3_creation.pdf")

    # -->
    p = time_series(3.835, 0.5, 1000)
    p = plot!(xlim=(900, 1000))
    savefig("../Slides/imgs/cycle_3_time_series.pdf")

    # -->
    f(x) = logistic_map(x, 3.8275)
    f³(x) = f.(f.(f.(x)))

    p = cobweb(f³, 0.19, 30)
    savefig("../Slides/imgs/cycle_3_intermittency.pdf")

    p = cobweb(f³, 0.19, 30)
    p = plot!(xlim=(0.48, 0.53), ylim=(0.5, 0.55))
    savefig("../Slides/imgs/cycle_3_intermittency_close_up.pdf")

    p = time_series(3.8282, 0.19, 1000)
    p = plot!(xlim=(20, 200))
    savefig("../Slides/imgs/cycle_3_intermittency_time_series.pdf")

    return p
end
