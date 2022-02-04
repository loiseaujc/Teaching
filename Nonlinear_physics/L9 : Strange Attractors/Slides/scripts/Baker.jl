using Plots

struct Point
    x :: Float64
    y :: Float64
end


function ℬ(p::Point ; a=1/3)

    # --> Extract the coordinates.
    x, y = p.x, p.y

    # --> Baker map.
    if x < 1/2
        x, y = 2*x, a*y
    else
        x, y = 2*x-1, 1 - a + a*y
    end

    return Point(x, y)
end

function plot_baker_map(P ; xlims=(0, 1), ylims=(0, 1))

    # --> Get data into the correct type.
    data = reduce(hcat, [[p.x, p.y] for p ∈ P])

    # --> Plot figure.
    p = scatter(
        data[1, :], data[2, :],
        markershape=:square,
        markersize = 0.05,
        markercolor=:black,
        legend=:none,
        xlims=xlims,
        ylims=ylims,
        aspect_ratio=:equal,
        axis=([], false),
        size=(512, 512),
    )

    return p
end

maxpts = 1_000_000
R = [Point(rand(2)...) for i = 1:maxpts]

p = plot_baker_map(R)
display(p)
savefig("Baker_map.png")

for i = 1:10
    # --> Iterate the map.
    R = ℬ.(R)

    # --> Plot the figure.
    p = plot_baker_map(R)
    display(p)
    savefig("Baker_map_$i.png")

end

p = plot_baker_map(R ; xlims=(0, 1/3), ylims=(0, 1/3))
savefig("Baker_map_zoom.png")

p = plot_baker_map(R ; xlims=(0, 1/9), ylims=(0, 1/9))
savefig("Baker_map_zoom.png")

p = plot_baker_map(R ; xlims=(0, 1/9), ylims=(0, 1/9))

savefig("Baker_map_zoom_bis.png")
