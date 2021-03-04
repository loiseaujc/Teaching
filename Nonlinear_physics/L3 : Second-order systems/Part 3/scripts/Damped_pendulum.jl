using DifferentialEquations
using Plots
using DelimitedFiles
using LinearAlgebra

function pendulum(du, u, p, t)

    ddu = -0.5*du -sin.(u)

    return ddu
end


u, du = π-0.01, 0.0
tspan = (0.0, 30.0)
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot(sol, vars=(1, 2))

outfile = "traj1ter.txt"
writedlm(outfile, sol.u/2)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)

u, du = -π+0.01, 0.0
prob = SecondOrderODEProblem(pendulum, du, u, tspan)
sol = solve(prob)

plot!(sol, vars=(1, 2))

outfile = "traj2ter.txt"
writedlm(outfile, sol.u/2)
u = readdlm(outfile)[:, end:-1:1]
writedlm(outfile, u)
