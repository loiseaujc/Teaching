# -->
using DifferentialEquations

# -->
using LinearAlgebra

# -->
using Plots; pyplot()
using LaTeXStrings

# upscale = 8 #8x upscaling in resolution
# fntsm = Plots.font("sans-serif", pointsize=round(10.0*upscale))
# fntlg = Plots.font("sans-serif", pointsize=round(14.0*upscale))
# default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)
# default(size=(800*upscale,200*upscale)) #Plot canvas size
# default(dpi=300) #Only for PyPlot - presently broken

# -->
function harmonic_oscillator!(du, u, p, t)

    """Definition of the harmonic oscillator."""

    # --> Unpack parameters.
    k, ω₀ = p

    # --> Equations of motion.
    du[1] = u[2]
    du[2] = -2k * u[2] - ω₀^2 * u[1]

    return du
end

function main()

    # -->
    tspan = (0.0, 25.0)
    x₀ = [1.0, 0.0]

    #############################################
    #####     Underdamped configuration     #####
    #############################################

    # --> Parameters.
    k, ω₀ = 0.25, 1.0
    params = (k, ω₀)

    # --> Define the problem.
    underdamped_oscillator = ODEProblem(harmonic_oscillator!, x₀, tspan, p=params)

    # --> Simulated the problem.
    underdamped_sol = solve(underdamped_oscillator, Tsit5())

    ############################################
    #####     Critically configuration     #####
    ############################################

    # --> Parameters.
    k, ω₀ = 1.0, 1.0
    params = (k, ω₀)

    # --> Define the problem.
    critdamped_oscillator = ODEProblem(harmonic_oscillator!, x₀, tspan, p=params)

    # --> Simulated the problem.
    critdamped_sol = solve(critdamped_oscillator, Tsit5())

    ############################################
    #####     Overdamped configuration     #####
    ############################################

    # --> Parameters.
    k, ω₀ = 1.25, 1.0
    params = (k, ω₀)

    # --> Define the problem.
    overdamped_oscillator = ODEProblem(harmonic_oscillator!, x₀, tspan, p=params)

    # --> Simulated the problem.
    overdamped_sol = solve(overdamped_oscillator, Tsit5())

    #####
    #####
    #####

    p = plot(
        underdamped_sol, vars=(0, 1),
        color=:chartreuse,
        label="Underdamped",
    )

    # p = plot!(
    #     critdamped_sol, vars=(0, 1),
    #     color=2,
    #     label="Critically damped"
    # )

    p = plot!(
        overdamped_sol, vars=(0, 1),
        color=:dimgray,
        linestyle=:dash,
        label="Overdamped",
        ylabel=L"\theta(t)",
        xlabel="Time",
        size=(1200, 300),
        framestyle=:box,
        tickfont=font(24),
        guidefont=font(28),
        ylim=(-0.5, 1.0),
        legendfont=font(16),
    )

    png("../Slides/imgs/harmonic_oscillator_regimes")

    return p
end
