using Plots; gr()
using LaTeXStrings

function logistic_map(x, μ)
    return μ * x .* ( 1 .- x )
end

function sine_map(x, μ)
    return μ * sin.(π * x)
end

function tent_map(x, μ)
    if 0 <= x <= 1/2
        return μ * x
    elseif 1/2 <= x <= 1
        return μ .- μ * x
    end
end

function cobweb(f, x₀, μ, n)

    # -->
    x = zeros(n)
    x[1] = x₀
    for i in 2:n
        x[i] = f(x[i-1], μ)
    end

    # -->
    y = collect(0:0.001:1)

    # -->
    p = plot(
        y, f.(y, μ),
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:none,
        color="red",
        size=(256, 256),
        aspect_ratio=:equal,
        framestyle=:box,
        linewidth=2,
        xlabel=L"x",
        ylabel=L"f(x)"
        )

    p = plot!(
        y, y,
        color="dimgray",
        linewidth=2,
    )

    i = 0
    p = plot!(
        [x[i+1], x[i+1]], [0, f(x[i+1], μ)],
        color="black",
        linewidth=1,
        arrow=true,
        )
    p = plot!(
        [x[i+1], f(x[i+1], μ)], [f(x[i+1], μ), f(x[i+1], μ)],
        color="black",
        linewidth=1,
        arrow=true,
        )

    for i in 1:(n-1)
        p = plot!(
            [x[i+1], x[i+1]], [x[i+1], f(x[i+1], μ)],
            color="black",
            linewidth=1,
            arrow=true,
            )
        p = plot!(
            [x[i+1], f(x[i+1], μ)], [f(x[i+1], μ), f(x[i+1], μ)],
            color="black",
            linewidth=1,
            arrow=true,
            )
    end

    return p
end

function main(x₀, μ, n)

    # -->
    p = cobweb(logistic_map, x₀, μ, n)

    # -->
    savefig("../Slides/imgs/cobweb_logistic_map.pdf")

    return p
end
