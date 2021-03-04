using DifferentialEquations
using Plots
using DelimitedFiles
using LinearAlgebra

function lotka_volterra(du, u, p, t)

    x, y = u

    du[1] = x*(3-x) - 2*x*y
    du[2] = y*(2-y) - x*y

    return
end

function lotka_volterra_bis(du, u, p, t)

    x, y = u

    du[1] = x*(3-x) - 2*x*y
    du[2] = y*(2-y) - x*y

    du .= -du

    return
end

tspan = (0.0, 20.0)
u₀ = [0.01, 0.01]
prob = ODEProblem(lotka_volterra, u₀, tspan)
sol = solve(prob)

outfile = "traj1.txt"
writedlm(outfile, sol.u)

u₀ = [0.001, 0.025]
prob = ODEProblem(lotka_volterra, u₀, tspan)
sol = solve(prob)

outfile = "traj2.txt"
writedlm(outfile, sol.u)

u₀ = [3, 3]
prob = ODEProblem(lotka_volterra, u₀, tspan)
sol = solve(prob)

outfile = "traj3.txt"
writedlm(outfile, sol.u)


u₀ = [4, 2]
prob = ODEProblem(lotka_volterra, u₀, tspan)
sol = solve(prob)

outfile = "traj4.txt"
writedlm(outfile, sol.u)

J = [-1 -2 ; -1 -1]
v = eigvecs(J)
ϵ = 1e-3

tspan = (0.0, 100.0)
u₀ = [1, 1] + ϵ*v[:, 2]
prob = ODEProblem(lotka_volterra, u₀, tspan)
sol = solve(prob)

outfile = "traj5.txt"
writedlm(outfile, sol.u)

plot(sol, vars=(1, 2))

tspan = (0.0, 100.0)
u₀ = [1, 1] - ϵ*v[:, 2]
prob = ODEProblem(lotka_volterra, u₀, tspan)
sol = solve(prob)

outfile = "traj6.txt"
writedlm(outfile, sol.u)
plot!(sol, vars=(1, 2))

tspan = (0.0, 5.0)
u₀ = [1, 1] - ϵ*v[:, 1]
prob = ODEProblem(lotka_volterra_bis, u₀, tspan)
sol = solve(prob)

outfile = "traj7.txt"
writedlm(outfile, sol.u)

plot(sol, vars=(1, 2), xlims=(0, 4), ylims=(0, 3))

tspan = (0.0, 2.85)
u₀ = [1, 1] + ϵ*v[:, 1]
prob = ODEProblem(lotka_volterra_bis, u₀, tspan)
sol = solve(prob)

outfile = "traj8.txt"
writedlm(outfile, sol.u)

plot(sol, vars=(1, 2), xlims=(0, 4), ylims=(0, 3))
