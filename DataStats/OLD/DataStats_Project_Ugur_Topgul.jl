### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 1d0a34e6-6d29-11eb-1d64-33fa134b4560
begin
	using PlutoUI
	using MAT
	using LinearAlgebra, Statistics
	using Plots
end

# ╔═╡ bfc84fdc-6d25-11eb-1d13-a909ac45e366
md"
# Project : Reduced-order modeling of the two-dimensional cylinder flow

The two-dimensional cylinder flow is a canonical of bluff body flows in fluid dynamics.
Despite its simplicity, this two-dimensional flow captures some fundamental features of larger scale flows as shown in the image below.

![](https://cdn.iflscience.com/images/091d5090-e168-5128-ba1f-8a06113f54c7/large-1464354324-40-von-karman-vortex-streets.jpg)

This flow pattern is known as the [Bénard-von Kàrmàn vortex street](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street) and corresponds to the periodic shedding of vortices from the cylinder (or the island in this image above).

From a mathematical point of view, the dynamics of this flow can be modeled by the incompressible Navier-Stokes equations

```math
\begin{aligned}
	\dfrac{\partial \mathbf{u}}{\partial t} + \nabla \cdot \left( \mathbf{u} \otimes \mathbf{u} \right) & = - \nabla p + \frac{1}{Re} \nabla^2 \mathbf{u} \\
\nabla \cdot \mathbf{u} & = 0
\end{aligned}
```

where $\mathbf{u}$ is the two-dimensional velocity field, $p$ the pressure field and $Re$ the Reynolds number (i.e. the ratio of the inertial and viscous forces).
In order to simulate these equations on a computer, they are typically discretized on a mesh with $n$ points such that $\mathbf{u} \in \mathbb{R}^{2n}$ and $p \in \mathbb{R}^n$ where $n$ can be of the order of several hundred thousands.
From a dynamical point of view however, the dynamics are rather simple and correspond to simple periodic dynamics.
As such, dynamically, only two degrees of freedom (e.g. the amplitude and the phase of the oscillation) are needed to characterize the state of the system rather than $n$ degrees.
In this notebook, you will have to analyze these dynamics using the different tools discussed during the class (i.e. dimensionality reduction using PCA or DMD, sparse sensor placement, least-squares and compressive sensing).
For that purpose, you'll be given a dataset consisting in snapshots of the vorticity field (i.e. $\omega = \nabla \times \mathbf{u}$) over time obtained by direct numerical simulation of the Navier-Stokes equations.
A random selection of these snapshots is shown below.
"

# ╔═╡ 1b3a546e-6d29-11eb-04fe-0f4201be3100
md"Dataset to be loaded : $@bind dset TextField()"

# ╔═╡ 0801bd0a-6d2a-11eb-0071-fdd4acaf308f
begin
	# --> Load the dataset.
	data = matread(dset)
	
	# --> Extract the mesh and the snapshots collection.
	mesh, X = [data["x"][:], data["y"][:]], data["data"]
end;

# ╔═╡ d19dff42-6d4e-11eb-19ec-ad9425b3d4c1
md"Display random snapshots from the database : $@bind go Button()"

# ╔═╡ 6a05539a-6d2a-11eb-1772-a76e67e6304e
md"
## Dimensionality reduction

As a starting point, let us try to determine the intrinsic dimensionality of our dataset.
For that purpose, we'll use *principal component analysis* (PCA).
A brief recap' of PCA is given below.

###### Principal Component Analysis

PCA is closely related to the SVD matrix factorization.
In this notebook, we will however use PCA through its correlation matrix formulation.
Given a data matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ where each column corresponds to one snapshot of the cylinder flow simulation with $m$ state variables and we have $n$ such snapshots sampled uniformly in time, let us first compute the mean flow, i.e.

```math
\overline{\mathbf{x}} = \dfrac{1}{n} \sum_{i=1}^n \mathbf{x}_i.
```

In a second step, the temporal correlation matrix

```math
	\boldsymbol{\Sigma}_{\mathbf{X}} = \dfrac{1}{n-1} \sum_{i=1}^n \left( \mathbf{x}_i - \overline{\mathbf{x}} \right)^T \left( \mathbf{x}_i - \overline{\mathbf{x}} \right)
```

is constructed.
Assuming that the mean flow has already been subtracted from the columns of $\mathbf{X}$ (` X .-= x̄` in Julia), this correlation matrix can easily be computed in one go as follows

```math
\boldsymbol{\Sigma}_{\mathbf{X}} = \dfrac{1}{n-1} \mathbf{X}^T \mathbf{X}.
```

In a third step, the eigendecomposition of $\boldsymbol{\Sigma}_{\mathbf{X}}$ is computed, i.e.

```math
	\boldsymbol{\Sigma}_{\mathbf{X}} \mathbf{V} = \mathbf{V} \boldsymbol{\Lambda}
```

where $\boldsymbol{\Lambda}$ is the diagonal matrix of eigenvalues and $\mathbf{V}$ is the corresponding matrix of eigenvectors normalized such that $\mathbf{V}^T \mathbf{V} = \mathbf{I}$.
The PCA modes can then be computed as

```math
\mathbf{U} = \dfrac{1}{\sqrt{N-1}} \mathbf{XV} \boldsymbol{\Lambda}^{-\frac{1}{2}}
```

while the projection of the snapshots into this particular basis is given by

```math
\mathbf{a} = \mathbf{U}^T \mathbf{X}.
```

### Application to the cylinder flow

It is now up to you.
Given the dataset already uploaded in the notebook as `X`, use the cells below to :

1. Compute the mean flow `x̄` and plot it using the `plot_flow_field(mesh, x̄)` command.
2. Compute the covariance matrix `Σ` and plot it using `heatmap(Σ)`. How do you interpret this plot?
3. Compute the eigendecomposition of `Σ` using `eigen(Σ)` and plot the distribution of the eigenvalues. Given that each eigenvalue is directly related to the amount of turbulent kinetic energy captured by the corresponding PCA mode, how many modes do you need to keep in your analysis to capture 99% of the kinetic energy ?
4. Compute and plot the first four PCA modes using `plot_flow_field(mesh, U[:, i])`. Eventhough you may not be familiar with fluid dynamics, try to explain the patterns you observe.
"

# ╔═╡ 18d3d7fe-83ef-11eb-38af-2ddbf81fca5d
mesh

# ╔═╡ 3ab18702-83ec-11eb-3ede-c1ac07a681ef
m,n=size(X)

# ╔═╡ e533f59e-6d2a-11eb-2fde-718f3c5041bd
# --> Compute the mean flow.
X̄=mean(X,dims=2)

# ╔═╡ d9c8eb26-6d37-11eb-2718-472a8ec69bd1
begin
	# --> Subtract mean from the data matrix.
	
	x_diff=(X.-X̄)
	
	# --> Compute the covariance matrix.
	
	Σ=(x_diff'*x_diff)/(n-1)
	
	# --> Plot the covariance matrix.
	
	heatmap(Σ[:,:],yflip=true)

end

# ╔═╡ d52ccb44-6d39-11eb-3dff-af3ddb563f11
md"
 **Interpration :**

	Heatmap plotting represents, Covariance is a concept that measures the variability of the linear relationship between two variables. If the result is positive (yellow colour), it indicates an increasing linear relationship, a negative (black colour) one indicates a decreasing linear relationship, and being around zero indicates that there is no relationship.
"

# ╔═╡ d5135362-6d39-11eb-1303-1b30121d1717
begin
	# --> Compute the eigendecomposition of Σ.
	
	Λ,V=eigen(Σ)
	Λ,V=Λ[end:-1:1],V[:,end:-1:1]
	
	# --> Plot the distribution of eigenvalues.
	
	plot(Λ)
	
end

# ╔═╡ c2efc92a-8514-11eb-289e-116a3c60852f
md"
 **Mode Guess :**

	You can see in the eigenvalues distribution graph, Approximately first 10 modes are capturing 99% of kinetic energy.
"

# ╔═╡ 9b0738cc-84c3-11eb-3657-c71cd3b6ccdd
ΛDiagonal=Diagonal(Λ)

# ╔═╡ e014a80a-84c3-11eb-363d-0d0a4e850570
Λ_for_PCA=ΛDiagonal[1:20,1:20]

# ╔═╡ fb09bb78-84c3-11eb-2c4d-81440681933f
Λ_cal=Matrix(I,20,20)/Λ_for_PCA^1/2

# ╔═╡ d4f6f550-6d39-11eb-1463-770f6b0a0742
# --> Compute the PCA modes.
	U=(1/√(n-1))*X*V[:,1:20]*Λ_cal

# ╔═╡ 05028eee-6d3a-11eb-3fbd-a75908821bb1
md"
**Pattern :**

	1- Laminar Vortex Shedding 

	2- Laminar Vortex Shedding 

	3- Vortex Shedding in Unsteady Flow

	4- Vortex Shedding in Unsteady Flow
"

# ╔═╡ 084a5980-6d3a-11eb-0c96-7d2cf3cec629
md"
## Sparse sensor placement

Measuring the velocity in the whole domain is technically impossible outside of a simulation.
In any experiment, information is gathered from limited sensor measurements.
Having gained some insight about the coherent structures existing in the flow, let us leverage these to guide us in placing actual sensors :
1. Choose a desired rank `r` for the PCA truncation (i.e. `Ψ = U[:, 1:r]`)
2. Compute the QR decomposition with pivot of `transpose(Ψ)` using `qr(transpose(Ψ), Val(true))` and return the `r` first pivots `p`.
3. Use the command `plot_flow_field(mesh, U[:, 1], p)` to superimpose the leading PCA mode and the location of the selected sensors.
4. Using your physical intuition, do these sensor locations make sense ? If so, why ? If not, why ?
"

# ╔═╡ d7cfb6c4-6d3c-11eb-0e83-e744df9b16c5
md"Rank `r` of the PCA basis : $@bind r Slider(1:20, default=2, show_value=true)"

# ╔═╡ 712547fe-6d3c-11eb-3b53-1982fe7051ec
# --> Truncated PCA basis Ψ (\Psi <TAB> to have Ψ displayed as variable name).
	Ψ = U[:, 1:r]

# ╔═╡ 7dce6f62-6d3c-11eb-061d-654eb12763b5
# --> Compute the QR decomposition with pivot.
begin
	Q,R,p=qr(transpose(Ψ),Val(true))
	r, p
end

# ╔═╡ 556fc7d2-851a-11eb-3a65-7b5adb389d91


# ╔═╡ c1c7fbc2-8518-11eb-0a8e-01d0c46ea141
md"
**Superimposed Sensor Locations :**
	
	Yes, It makes sense because If we increase the number of sensors, we will get more accurate results.
"

# ╔═╡ 9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
md"
Denoting by `a` the projection `tranpose(Ψ) * X` (i.e. the projection of the data into the leading PCA subspace), `y = X[p, :]` the measurements and `Θ = Ψ[p, :]` the measurement matrix, use the least-squares technique to solve

```math
 \mathbf{y} = \boldsymbol{\Theta} \hat{\mathbf{a}}
```

and compare the the time-series of the estimated `â` coefficients with the true ones in `a` for varying values of `r`.
What do you observe ?

"

# ╔═╡ b60d169a-851e-11eb-2d48-3dee2fcf7d52
md"
**Answer :**

	When we plot a and â matrices to compare, after using least-square technique (â) graph is getting more clear to understand. 
	"

# ╔═╡ 353766a4-6d3d-11eb-2aa6-f3683b7d2d19
begin
	# --> Takes the measurements y.
	
	y = X[p, :]
	
	# --> Build the measurement matrix Θ (\Theta <TAB>)
	
	Θ = Ψ[p, :]
	
	
	# --> Obtain the true low-dimensional projection (a = transpose(Ψ) * X).
	
	a = transpose(Ψ) * X

end

# ╔═╡ 34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# --> Solve Θ * â = y.
begin
	â=Θ\y
end

# ╔═╡ 5c34e7d2-851f-11eb-08c1-d1738f2d6082
plot(a')

# ╔═╡ 71b5a6e2-850a-11eb-0b23-fdb74d664220
begin
	plot(1:n,â')
end

# ╔═╡ 71109254-6d3d-11eb-2e41-71bf5c1543f9
md"
**What do you observe as `r` increases ?`**

	If rank increases, linearly independent rows or columns are increasing. Then rank indicates the intactness of a system. Then you can see changes in low-dimensional projection's graph. First two rank have more higher amplitudes but rest of number of rank graphs very similar amplitudes to each other.  
"

# ╔═╡ 16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
md"
## Inferring the vorticity field from sparse measurements

Now that we know where to place our sensors and can reconstruct the low-dimensional state vector $\mathbf{a}$, let us recontruct the whole vorticity field.
Denoting by $\hat{\mathbf{a}}$ the estimated low-dimensional state vector, the whole flow field can be reconstructed as

```math
\hat{\omega}(\mathbf{x}, t_k) = \boldsymbol{\Psi} \hat{\mathbf{a}}.
```

The reconstruction error is defined as

```math
\mathrm{Err} = \dfrac{1}{N} \sum_{i=1}^N \| \hat{\omega}_i - \omega_i \|_2^2
```

Write a function that computes the reconstruction error as a function of the number $r$ of sensors place in the flow.
"

# ╔═╡ 16825226-7a6e-11eb-23e3-0baf0bad7292
ω̂=Ψ[1:r,:]*â[:,1:r]

# ╔═╡ 1679e8ca-7a6e-11eb-2ca3-418d0ca74765
md"
Based on your results, how would choose $r$ such that it best balances the reconstruction accuracy and the number of required sensors ?
Justify.
"

# ╔═╡ 953d2ee4-8525-11eb-17c8-2dae551f2838
ω=Ψ[1:r,:]*a[:,1:r]

# ╔═╡ ba59f5ea-8525-11eb-09bd-4f4069ebe3cf
ω_diff_norm=(norm(ω̂-ω,2))^2

# ╔═╡ 9441ea26-850b-11eb-365c-6fcb758f0683
err=1/n*ω_diff_norm

# ╔═╡ 167978e0-7a6e-11eb-334a-5b057dc68025
md"
**Answer :**

	If we increase r value, we can get lower error and high accuracy. In this case, the error is approximately 1.35e-8. So, this error value is too small. For that reason, we shouldn't increase or decrease the r value.
"

# ╔═╡ 163c4aba-7a6e-11eb-14c7-835023e4773a
md"
The methodology presented here is quite general and can be applicable to large variety of different engineering fields.
Suggest three applications of this methodology to other problems and explain why you think if would be a good way to tackle the problem.
"

# ╔═╡ ff9de946-7a6e-11eb-3360-0f66704f508a
md"
**Answer :**

This methodology also using for
	
**Neuroscience :**

	PCA method is calling spike-triggered covariance analysis. This method is used in neuroscience to identification of the specific properties that increase a neuron's probability of generating an action potential and using for white noise process , as a stimulus (usually either as a sensory input to a test subject, or as a current injected directly into the neuron) and records a train of action potentials, or spikes, produced by the neuron as a result.

**Quantitative Finance :**

	In quantitative finance, PCA can be applied to the risk management of interest rate derivative portfolios. Converting risks to be represented as those to factor loadings (or multipliers) provides assessments and understanding beyond that available to simply collectively viewing risks.

**Medical Data Correlation :**

	If I give an example to Medical Data Correlation:
	PCA has been used to show correlation of Cholesterol with low density lipo-protein.


"

# ╔═╡ ff8b5d8c-7a6e-11eb-3f98-51f763ba3743


# ╔═╡ ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5


# ╔═╡ ff55d27a-7a6e-11eb-2a81-5feaf3b230c6


# ╔═╡ 72675b82-6d4d-11eb-352e-8b1d5824f865
function circle(xc, yc, r)
	θ = LinRange(0, 2π, 128)
	return xc .+ r*sin.(θ), yc .+ r*cos.(θ)
end

# ╔═╡ 727ad7ca-6d4d-11eb-15c5-bf6a4989e458
function plot_flow_field(mesh, x)
	
	scale = maximum(abs.(x))
	
	p = heatmap(
		mesh[1], mesh[2], reshape(x, length(mesh[2]), length(mesh[1])),
		axis=[],
		ylims=(-2.5, 2.5),
		xlims=(-2.5, 10),
		aspect_ratio=1,
		legend=false,
		c=cgrad(:bwr),
		clims=(-scale, scale),
	)
	
	plot!(circle(0, 0, 0.5), seriestype=[:shape], lw=2, c=:lightgray)
	
	return p
end

# ╔═╡ 724d6cfe-6d4d-11eb-0596-b9c552a25809
function plot_flow_field(mesh, x, pxl)
	
	# --> Plot the flow field.
	p = plot_flow_field(mesh, x)
	
	# --> Get the cartesian indices.
	idx = CartesianIndices((1:length(mesh[2]), 1:length(mesh[1])))
	
	ix = [mesh[1][idx[i][2]] for i in pxl]
	iy = [mesh[2][idx[i][1]] for i in pxl]
	
	scatter!(ix, iy, ms=2)
	
	return p
end

# ╔═╡ 6a1ee816-6d2a-11eb-026c-890add109c54
begin
	go
	plot(
		plot_flow_field(mesh, X[:, rand(1:400)]),
		plot_flow_field(mesh, X[:, rand(1:400)]),
		plot_flow_field(mesh, X[:, rand(1:400)]),
		plot_flow_field(mesh, X[:, rand(1:400)]),
		layout=(2, 2),
		size=(512, 256),
		)
end


# ╔═╡ f4094c04-6d39-11eb-1930-457bdb436549
# --> Plot PCA 1.
	plot_flow_field(mesh, U[:,1])

# ╔═╡ f7dadaa0-6d39-11eb-041b-396f59ed4ed8
# --> Plot PCA 2.
plot_flow_field(mesh, U[:, 2])

# ╔═╡ fba0f79e-6d39-11eb-1506-c7a6a40ca7ee
# --> Plot PCA 3.
plot_flow_field(mesh, U[:, 3])

# ╔═╡ fb88bc94-6d39-11eb-2591-293bff72a732
# --> Plot PCA 4.
plot_flow_field(mesh, U[:, 4])

# ╔═╡ 7db635dc-6d3c-11eb-2c41-218475c90ffe
# --> Plot the first PCA mode superimposed with the sensor locations.
plot_flow_field(mesh, U[:,1],p)

# ╔═╡ Cell order:
# ╠═1d0a34e6-6d29-11eb-1d64-33fa134b4560
# ╟─bfc84fdc-6d25-11eb-1d13-a909ac45e366
# ╠═1b3a546e-6d29-11eb-04fe-0f4201be3100
# ╠═0801bd0a-6d2a-11eb-0071-fdd4acaf308f
# ╠═d19dff42-6d4e-11eb-19ec-ad9425b3d4c1
# ╠═6a1ee816-6d2a-11eb-026c-890add109c54
# ╟─6a05539a-6d2a-11eb-1772-a76e67e6304e
# ╠═18d3d7fe-83ef-11eb-38af-2ddbf81fca5d
# ╠═3ab18702-83ec-11eb-3ede-c1ac07a681ef
# ╠═e533f59e-6d2a-11eb-2fde-718f3c5041bd
# ╠═d9c8eb26-6d37-11eb-2718-472a8ec69bd1
# ╠═d52ccb44-6d39-11eb-3dff-af3ddb563f11
# ╠═d5135362-6d39-11eb-1303-1b30121d1717
# ╟─c2efc92a-8514-11eb-289e-116a3c60852f
# ╠═9b0738cc-84c3-11eb-3657-c71cd3b6ccdd
# ╠═e014a80a-84c3-11eb-363d-0d0a4e850570
# ╠═fb09bb78-84c3-11eb-2c4d-81440681933f
# ╠═d4f6f550-6d39-11eb-1463-770f6b0a0742
# ╠═f4094c04-6d39-11eb-1930-457bdb436549
# ╠═f7dadaa0-6d39-11eb-041b-396f59ed4ed8
# ╠═fba0f79e-6d39-11eb-1506-c7a6a40ca7ee
# ╠═fb88bc94-6d39-11eb-2591-293bff72a732
# ╟─05028eee-6d3a-11eb-3fbd-a75908821bb1
# ╟─084a5980-6d3a-11eb-0c96-7d2cf3cec629
# ╠═d7cfb6c4-6d3c-11eb-0e83-e744df9b16c5
# ╠═712547fe-6d3c-11eb-3b53-1982fe7051ec
# ╠═7dce6f62-6d3c-11eb-061d-654eb12763b5
# ╠═7db635dc-6d3c-11eb-2c41-218475c90ffe
# ╟─556fc7d2-851a-11eb-3a65-7b5adb389d91
# ╟─c1c7fbc2-8518-11eb-0a8e-01d0c46ea141
# ╟─9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
# ╟─b60d169a-851e-11eb-2d48-3dee2fcf7d52
# ╠═353766a4-6d3d-11eb-2aa6-f3683b7d2d19
# ╠═34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# ╠═5c34e7d2-851f-11eb-08c1-d1738f2d6082
# ╠═71b5a6e2-850a-11eb-0b23-fdb74d664220
# ╟─71109254-6d3d-11eb-2e41-71bf5c1543f9
# ╟─16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
# ╠═16825226-7a6e-11eb-23e3-0baf0bad7292
# ╟─1679e8ca-7a6e-11eb-2ca3-418d0ca74765
# ╠═953d2ee4-8525-11eb-17c8-2dae551f2838
# ╠═ba59f5ea-8525-11eb-09bd-4f4069ebe3cf
# ╠═9441ea26-850b-11eb-365c-6fcb758f0683
# ╟─167978e0-7a6e-11eb-334a-5b057dc68025
# ╟─163c4aba-7a6e-11eb-14c7-835023e4773a
# ╠═ff9de946-7a6e-11eb-3360-0f66704f508a
# ╟─ff8b5d8c-7a6e-11eb-3f98-51f763ba3743
# ╟─ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5
# ╟─ff55d27a-7a6e-11eb-2a81-5feaf3b230c6
# ╟─727ad7ca-6d4d-11eb-15c5-bf6a4989e458
# ╟─72675b82-6d4d-11eb-352e-8b1d5824f865
# ╟─724d6cfe-6d4d-11eb-0596-b9c552a25809
