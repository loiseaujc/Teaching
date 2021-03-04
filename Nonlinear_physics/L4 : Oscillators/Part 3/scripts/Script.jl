using Plots
using DelimitedFiles
using DifferentialEquations
using FFTW
using Statistics

function van_der_pol(du, u, p, t)

    # --> Parameter.
    ϵ = p

    # --> Governing equations.
    ddu = -u + ϵ*(1 - u^2) * du

    return ddu
end

#####
#####
#####

ps = 0:0.25:2

# --> Initial condition.
x₀ = 2.0
tspan = (0, 100.0)

plot()
for (i, p) in enumerate(ps)
    prob = SecondOrderODEProblem(van_der_pol, 0.0, x₀, tspan, p)
    sol = solve(prob, saveat=0.01)
    plot!(sol, vars=(2, 1), aspect_ratio=:equal)

    outfile = "../van_der_pol_traj_$i.txt"
    writedlm(outfile, sol.u)
    u = readdlm(outfile)[end÷2:end, end:-1:1] ./ 2
    writedlm(outfile, u)

end

plot!()

#####
#####
#####

x₀ = 0.26465863600724265
tspan = (0.0, 100.0)
p = 0.1

prob = SecondOrderODEProblem(van_der_pol, 2.0031780194286033, x₀, tspan, p)
sol = solve(prob, saveat=0.01)

plot(sol, vars=(1, 2), color=:gray, aspect_ratio=:equal)
plot(sol, vars=(0, 2), color=:gray, aspect_ratio=:equal)

outfile = "../van_der_pol_traj.txt"
writedlm(outfile, sol.u)
u = readdlm(outfile)[end÷2:end, end:-1:1] ./ 2
writedlm(outfile, u)

#####
#####
#####

t = LinRange(0, 200, 2*4096)
ω = 2π .* fftfreq(length(t), 1/(t[2]-t[1]))
ω = fftshift(ω)

ps = 0:0.001:0.1

A = zeros(length(t), length(ps))

for (i, p) in enumerate(ps)
    prob = SecondOrderODEProblem(van_der_pol, 0.0, x₀, (t[1], t[end]), p)
    sol = solve(prob)
    u = sol(t)[2, :]
    u .-= mean(u)
    A[:, i] = log.(abs.(fftshift(fft(u))))
end

heatmap(
    ω, ps, transpose(A),
    xlims=(-5, 5),
    clims=(2, maximum(A)),
    legend=false,
    xlabel="Frequency ω",
    ylabel="ϵ",
    size=(512, 256),
    guidefontsize=font(9),
    )

png("../imgs/van_der_pol_spectrum")



x₀ = 0.5
tspan = (0.0, 110.0)
p = 0.1

prob = SecondOrderODEProblem(van_der_pol, 0.0, x₀, tspan, p)
sol = solve(prob, saveat=0.1)

plot(sol, vars=(1, 2), color=:gray, aspect_ratio=:equal)

outfile = "../van_der_pol_traj_bis.txt"
writedlm(outfile, sol.u)
u = readdlm(outfile)[:, end:-1:1] ./ 2
writedlm(outfile, u)
