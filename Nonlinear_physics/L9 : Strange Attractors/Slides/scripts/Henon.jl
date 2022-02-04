using Plots

struct Point
    x :: Float64
    y :: Float64
end

d(p::Point) = p.x^2 + p.y^2

function ℋ(p::Point ; a=1.4, b=0.3)

    x = p.y + 1 - a*p.x^2
    y = b*p.x

    return Point(x, y)
end

maxpts = 2_000_000
R = [Point(0.1rand(2)...) for i =1:maxpts]

for i = 1:100
    # --> Iterate.
    R = ℋ.(R)

    # --> Filter points which diverged.

end

data = reduce(hcat, [[r.x, r.y] for r ∈ R])'

p = scatter(
    data[:, 1], data[:, 2],
    markershape=:square,
    markersize = 0.1,
    markercolor=:white,
    legend=:none,
    xlims=(-1.5, 1.5),
    ylims=(-0.5, 0.5),
    #aspect_ratio=:equal,
    axis=([], false),
    size=(512, 512),
    background_color=:black
)

savefig("Henon_map.png")

xlims!(0.5, 0.7)
ylims!(0.15, 0.21)
savefig("Henon_map_zoom.png")

xlims!(0.625, 0.64)
ylims!(0.187, 0.19)
savefig("Henon_map_zoom_bis.png")
