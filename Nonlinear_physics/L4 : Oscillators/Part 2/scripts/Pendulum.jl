using DifferentialEquations
using Plots
using DelimitedFiles
using LinearAlgebra

function pendulum(du, u, p, t)

    ddu = -sin.(u)

    return ddu
end

plot()
U = [π-0.01, 3π/4, π/2, π/4, π/8]
for (i, u) in enumerate(U)
    tspan = (0.0, 30.0)
    prob = SecondOrderODEProblem(pendulum, 0.0, u, tspan)
    sol = solve(prob, dt=0.1, alg=VerletLeapfrog())

    plot!(sol, vars=(1, 2), color=:gray)

    outfile = "pendulum_traj_$i.txt"
    writedlm(outfile, sol.u)
    u = readdlm(outfile)[:, end:-1:1]
    writedlm(outfile, u)

end

plot!(xlims=(-π, π))

#######

ω(ϵ) = 1 - (3ϵ/8)
x(t, ϵ) = cos(ω(ϵ)*t) + ϵ/32 * (cos(ω(ϵ)*t) - cos(3ω(ϵ)*t))
dx(t, ϵ) = -ω(ϵ) * sin(ω(ϵ)*t) + ϵ/32 * (-ω(ϵ) * sin(ω(ϵ)*t) + 3ω(ϵ) * sin(3ω(ϵ)*t))
θ(t, θ₀) = θ₀ * x(t, θ₀^2/6)
dθ(t, θ₀) = θ₀ * dx(t, θ₀^2/6)

harmonic(t, θ₀) = θ₀ * cos(t)
dharmonic(t, θ₀) = -θ₀ * sin(t)

t = LinRange(0, 15, 1501)

for (i, u) in enumerate(U[2:end])
    plot!(dθ.(t, u), θ.(t, u), linestyle=:dash, color=:red)

    outfile = "approx_traj_$i.txt"
    approx = hcat([θ.(t, u), dθ.(t, u)]...)
    @show size(approx)
    writedlm(outfile, approx)

    harm = hcat([harmonic.(t, u), dharmonic.(t, u)]...)
    plot!(dharmonic.(t, u), harmonic.(t, u), color=:blue)
    outfile = "harmonic_traj_$i.txt"
    writedlm(outfile, harm)

end

plot!(xlims=(-π, π))
