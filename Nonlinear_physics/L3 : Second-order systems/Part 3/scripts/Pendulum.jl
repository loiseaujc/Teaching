using DifferentialEquations
using Plots
using DelimitedFiles
using LinearAlgebra

function pendulum(du, u, p, t)

    ddu = -sin.(u)

    return ddu
end

u, du = π/4, 0.0
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot(sol, vars=(1, 2))

outfile = "traj1.txt"
writedlm(outfile, sol.u/π)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)


u, du = π/2, 0.0
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj2.txt"
writedlm(outfile, sol.u/π)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)

u, du = 3π/4, 0.0
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj3.txt"
writedlm(outfile, sol.u/π)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)

u, du = π-0.01, 0.0
tspan = (0.0, 40.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj4.txt"
writedlm(outfile, sol.u/π)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)







u, du = π/4, 0.0
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot(sol, vars=(1, 2))

outfile = "traj1bis.txt"
writedlm(outfile, sol.u/2)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)


u, du = π/2, 0.0
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj2bis.txt"
writedlm(outfile, sol.u/2)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)

u, du = π-0.01, 0.0
tspan = (0.0, 40.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj3bis.txt"
writedlm(outfile, sol.u/2)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)

u, du = -2π, 5π/6
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj4bis.txt"
writedlm(outfile, sol.u)
u = readdlm(outfile)[:, end:-1:1]
u = u[abs.(u[:, 1]) .< π, :]
writedlm(outfile, u/2)

u, du = 2π, -5π/6
tspan = (0.0, 10.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj5bis.txt"
writedlm(outfile, sol.u)
u = readdlm(outfile)[:, end:-1:1]
u = u[abs.(u[:, 1]) .< π, :]
writedlm(outfile, u/2)
