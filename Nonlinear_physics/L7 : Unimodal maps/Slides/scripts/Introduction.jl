using Plots; gr()
using LaTeXStrings

function main()

    # --> Logistic map.
    logistic_map(x, μ) = μ * x .* ( 1 .- x)

    # --> Sine map.
    sine_map(x, μ) = μ * sin.(π * x)

    # --> Tent map.
    function tent_map(x, μ)

        y = similar(x)
        y[x .< 1/2] .= μ * x[x .< 1/2]
        y[x .>= 1/2] .= μ .- μ * x[x .>= 1/2]

        return y
    end

    # -->
    x = collect(0:0.001:1)

    # -->
    p1 = plot(
        x, logistic_map(x, 3),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(256, 256),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"\mu x(1-x)"
        )
    p1 = plot!(
        x, x,
        color="dimgray",
        linewidth=2,
        )

    # -->
    p2 = plot(
        x, sine_map(x, 0.75),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(256, 256),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"\mu \sin( \pi x)"
    )

    p2 = plot!(
        x, x,
        color="dimgray",
        linewidth=2
    )

    # -->
    p3 = plot(
        x, tent_map(x, 1.5),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(256, 256),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"T(x, \mu)"
    )

    p3 = plot!(
        x, x,
        color="dimgray",
        linewidth=2
    )

    # -->
    p = plot(
        p1, p2, p3,
        layout=(1, 3),
        size=(256*3, 256),
    )

    # -->
    savefig("../Slides/imgs/1D_maps_introduction.pdf")

    return p
end
