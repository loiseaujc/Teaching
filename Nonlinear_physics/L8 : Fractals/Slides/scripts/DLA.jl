using Plots, DelimitedFiles

bernoulli(p) = rand() < p

############################################
#####                                  #####
#####     Define the Random Walker     #####
#####                                  #####
############################################

import Base: step, position

abstract type RandomWalker end

struct Walker2D <: RandomWalker
    x :: Int
    y :: Int
end

Walker2D() = Walker2D(0, 0)
Walker2D(pos::Tuple{Int, Int}) = Walker2D(pos[1], pos[2])

position(w::Walker2D) = (w.x, w.y)
step(w::Walker2D) = rand( ((1, 0), (0, 1), (-1, 0), (0, -1)) )
update(w::W) where {W <: RandomWalker} = W(position(w) .+ step(w))

# step(w::Walker2D) = (-1, rand( (-1, 1) ))
# update(w::W) where {W <: RandomWalker} = W(position(w) .+ step(w))

##########################################
#####                                #####
#####     Define the DLA process     #####
#####                                #####
##########################################

abstract type DLA end

struct VerticalDLA <: DLA
    width :: Int
    height :: Int
    stickiness :: Real
end

struct RadialDLA <: DLA
    width :: Int
    height :: Int
    stickiness :: Real
end

VerticalDLA(w, h) = VerticalDLA(w, h, 1)
RadialDLA(w, h) = RadialDLA(w, h, 1)

stop(dla::VerticalDLA, L) = sum(L[end-2, :]) > 0

function stop(dla::RadialDLA, L)
    at_boundaries = sum(L[:, 2]) + sum(L[:, end-2]) + sum(L[2, :]) + sum(L[end-2, :])
    return at_boundaries > 0
end

function init_lattice(dla::VerticalDLA)

    # --> Initial lattice.
    S = zeros(Bool, dla.height, dla.width)

    # --> Sets the seeds.
    S[1, :] .= true

    return S
end

function init_lattice(dla::RadialDLA)

    # --> Initial lattice.
    S = zeros(Bool, dla.height, dla.width)

    # --> Sets the seeds.
    S[dla.height ÷ 2, dla.width ÷ 2] = true

    return S
end

init_walker(dla::VerticalDLA) = Walker2D(dla.height-1, rand(1:dla.width))

function init_walker(dla::RadialDLA)

    # --> Choose boundary.
    n = rand(1:4)

    if n == 1
        w = Walker2D(2, rand(2:dla.width-1))
    elseif n == 2
        w = Walker2D(dla.height-1, rand(2:dla.width-1))
    elseif n == 3
        w = Walker2D(rand(2:dla.height-1), 2)
    else
        w = Walker2D(rand(2:dla.height-1), dla.width-1)
    end

    return w
end

function enforce_bc(dla::VerticalDLA, w::Walker2D)
    x, y = position(w)
    if w.x == dla.height
        x = dla.height-1
        y = rand(1:dla.width)
    end
    return Walker2D(x, mod1(y, dla.width))
end

function enforce_bc(dla::RadialDLA, w::Walker2D)

    x, y = position(w)

    if w.x == 1
        x = 2
    elseif w.x == dla.height
        x = dla.height-1
    end

    if w.y == 1
        y = 2
    elseif w.y == dla.width
        y = dla.width-1
    end

    return Walker2D(x, y)
end

function update_lattice(dla::T, L, walkers) where {T <: DLA}

    idx = []
    height, width = size(L)

    for (i, w) in enumerate(walkers)
        # --> Check if walker has touch the DLA structure.
        im, ip = mod1(w.x-1, height), mod1(w.x+1, height)
        jm, jp = mod1(w.y-1, width), mod1(w.y+1, width)

        sum_neighbours = L[im, w.y] + L[ip, w.y] + L[w.x, jm] + L[w.x, jp]
        sum_neighbours += L[im, jm] + L[im, jp] + L[ip, jm] + L[ip, jp]

        if (sum_neighbours > 0) && (bernoulli(dla.stickiness))
            L[w.x, w.y] = true
            push!(idx, i)
        end
    end

    # --> Remove the discarded walkers.
    splice!(walkers, sort(idx))

    return L, walkers
end

function simulate(dla::T ; nwalkers=200, maxiter=100) where {T <: DLA}

    # --> Initialize nwalkers.
    walkers = [init_walker(dla) for i = 1:nwalkers]

    # --> Initialize lattice.
    L = init_lattice(dla)

    for i = 1:maxiter

        # --> Generate new walkers if needed.
        while length(walkers) < nwalkers
            push!(walkers, init_walker(dla))
        end

        # --> Update all the walkers positions.
        # walkers = [enforce_bc(dla, update(w)) for w ∈ walkers]
        walkers = [update(w) for w ∈ walkers]

        # --> Enforce boundary conditions.
        walkers = [enforce_bc(dla, w) for w ∈ walkers]

        # --> Check if simulatation has to end.
        if stop(dla, L)
            break
        end

        # --> Update lattice with walkers that sticked
        #     and remove them from simluation.
        L, walkers = update_lattice(dla, L, walkers)
    end

    return L
end

function fractal_dimension(dla::RadialDLA, L)

    # -->
    width = collect(1:dla.width)
    height = collect(1:dla.height)

    x = getindex.(Iterators.product(width, height), 1)
    y = getindex.(Iterators.product(width, height), 2)

    x, y = x .- dla.width ÷ 2 , y .- dla.height ÷ 2

    R = sqrt.(x .^ 2 + y .^2)

    radius = LinRange(10, maximum(R), 32)

    mass = zero(radius)

    for (i, r) in enumerate(radius)
        mass[i] = sum(L[R .< r])
    end

    return radius, mass
end


function main()

    width, height = 512, 512

    dla = RadialDLA(width, height)

    L = simulate(dla ; maxiter=1_000_000_000, nwalkers=100)

    p = heatmap(L, axis=([], false), aspect_ratio=:equal, legend=:none, c=:binary, size=(512, 512))
    savefig("../imgs/DLA_1.png")
    L = simulate(dla ; maxiter=1_000_000_000, nwalkers=100)

    p = heatmap(L, axis=([], false), aspect_ratio=:equal, legend=:none, c=:binary, size=(512, 512))
    savefig("../imgs/DLA_2.png")

    L = simulate(dla ; maxiter=1_000_000_000, nwalkers=100)

    p = heatmap(L, axis=([], false), aspect_ratio=:equal, legend=:none, c=:binary, size=(512, 512))
    savefig("../imgs/DLA_3.png")

    p = heatmap(-(L .- 1), axis=([], false), aspect_ratio=:equal, legend=:none, c=:binary, size=(512, 512), background_color=:black)
    savefig("../imgs/DLA_4.png")

    r, m = fractal_dimension(dla, L)

    open("fractal_scaling.dat", "w") do io
        writedlm(io, [r m])
    end

    return r, m
end
