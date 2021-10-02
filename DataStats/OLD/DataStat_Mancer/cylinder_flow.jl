### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ ae0cca20-7a9d-11eb-1023-457ff1070586
begin
	import Pkg
	Pkg.add("MAT")
end

# ╔═╡ bfc84fdc-6d25-11eb-1d13-a909ac45e366
md"
# Project : Reduced-order modeling of the two-dimensional cylinder flow

The two-dimensional cylinder flow is a canonical of bluff body flows in fluid dynamics.
Despite its simplicity, this two-dimensional flow captures some fundamental features of larger scale flows as shown in the image below.

![](https://cdn.iflscience.com/images/091d5090-e168-5128-ba1f-8a06113f54c7/large-1464354324-40-von-karman-vortex-streets.jpg)

![](https://giphy.com/gifs/lU9KoIIYUoE088ilzI/html5)

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
#C:\Users\ASUS\Downloads\cylinder_dataset.mat

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

# ╔═╡ e533f59e-6d2a-11eb-2fde-718f3c5041bd
# --> Compute the mean flow.
begin
	m, n = size(X)
	sum = zeros(m)
	
	for i = 1:n
		sum .+= X[:,i] 
	end
	
	x̄ = sum./n
end

# ╔═╡ f808cd32-80e3-11eb-298b-750b651bf52f
size(x̄)

# ╔═╡ d9c8eb26-6d37-11eb-2718-472a8ec69bd1
begin
	# --> Subtract mean from the data matrix.
	
	for i=1:n
		X[:,i] .-= x̄
	end
	
	# --> Compute the covariance matrix.
	
	Σ = X'*X./(n-1)
	
	# --> Plot the covariance matrix.
	
	heatmap(Σ)

end

# ╔═╡ 6e724220-80e5-11eb-2143-25ab4008abd8
Σ

# ╔═╡ d52ccb44-6d39-11eb-3dff-af3ddb563f11
md"
 **Interpretation :**

The heatmap looks like an inversed tridiagonal matrix denoting the periodic nature of the data. We can notice that the direction shows that the covarience is negative since one variable decreases while the other increases.
"

# ╔═╡ d5135362-6d39-11eb-1303-1b30121d1717
begin
	# --> Compute the eigendecomposition of Σ.
	
	λp, Vp = eigen(Σ)

	λ = sort(λp, rev=true)
	V = Vp[:,n:-1:1]
	
	Λ = Diagonal(λ)
	
	# --> Plot the distribution of eigenvalues.
	
	plot(λ[1:50], marker=".",
	xlabel="time step",
	ylabel="λ",
	leg=:false,
	title="Eigenvalues distribution",
	)

end

# ╔═╡ 007fecfe-8525-11eb-1def-452921d156db
	Λs = Λ[1:100, 1:100]

# ╔═╡ 16e71820-8525-11eb-2d55-3bf6bdeb2c5b
	Λsq = Matrix(I,100,100)/Λs^(1/2)

# ╔═╡ 4f4aec50-8520-11eb-1152-495ece40e16b
begin
	ss = 0
	elmnt = 0
	for i=1:n
		ss += λ[i] 
		if ss < 2241.986528676953
			elmnt += 1
		end
	end
	elmnt
end

# ╔═╡ be077240-85a1-11eb-1a97-678b564c16fb
md"
Each eigenvalue being related to the amount of turbulent kinetic energy within its respective PCA mode, we would need 7 eigen modes to keep in the analysis in order to capture 99% of the kinetic energy. Here 99% represents approximately 2241.98 which is 99% of the sum of the eigenvalues.

"

# ╔═╡ d4f6f550-6d39-11eb-1463-770f6b0a0742
# --> Compute the PCA modes.

	U = (1/√(n-1))*X*V[:, 1:100]*Λsq


# ╔═╡ 05028eee-6d3a-11eb-3fbd-a75908821bb1
md"
**Pattern :**

We can notice that for the first PCA mode, the flow is regular along the bluff body which can be approximated to ideal conditions flow whereas in the second one, we observe a small perturbation. In the third PCA mode, smaller vortexes start appearing thus creating a cascade effect which intensifies in the fourth PCA mode. This phenomenon keeps on going until the injection scale (scale of the vortex of the first PCA mode) becomes a small scale which is the last step before the destruction of the vortex by dissipation.

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
md"Rank `r` of the PCA basis : $@bind r Slider(1:50, default=2, show_value=true)"

# ╔═╡ 712547fe-6d3c-11eb-3b53-1982fe7051ec
# --> Truncated PCA basis Ψ (\Psi <TAB> to have Ψ displayed as variable name).

	Ψ = U[:, 1:r]

# ╔═╡ 7dce6f62-6d3c-11eb-061d-654eb12763b5
# --> Compute the QR decomposition with pivot.

	Q, R, pp = qr(transpose(Ψ), Val(true))

# ╔═╡ 745985c0-851a-11eb-21ca-df99ff11c72d
	p = pp[1:r]

# ╔═╡ ffe4b9d0-85a5-11eb-12b6-ebcd30ecf3be
md"
**Question 4**

At first, the location of the sensors seem pretty random, but then by increasing the number of sensors, it seems that they are located in a way that they capture information at each position in the vortex scale. After that we can assume that the information is similar thanks to the periodic pattern observed.

The gif, joined to this document, shows the change in the position of the sensors with the increase of r.

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

# ╔═╡ 353766a4-6d3d-11eb-2aa6-f3683b7d2d19
begin
	# --> Takes the measurements y.
	y = X[p, :]
	
	# --> Build the measurement matrix Θ (\Theta <TAB>)
	Θ = Ψ[p, :]
	
	# --> Obtain the true low-dimensional projection (a = transpose(Ψ) * X).
	#a = transpose(Ψ) * X
	a = Ψ' * X

end

# ╔═╡ 34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# --> Solve Θ * â = y.
	â = qr(Θ) \ y
	

# ╔═╡ c77592e0-8519-11eb-0b1d-7974089a12b6
y

# ╔═╡ 6be4c214-6d3d-11eb-1113-598d240bc3c0
begin
	# --> Plot the time series.
	
	plot(â[1,1:50], marker=".",
	xlabel="time step",
	ylabel="â",
	label="â1",
	leg=:true,
	title="Time series",
	size=(650, 350)
	)
	plot!(â[2,1:50], marker=".", label="â2")
	plot!(â[3,1:50], marker=".", label="â3")
	plot!(â[4,1:50], marker=".", label="â4")
	plot!(â[5,1:50], marker=".", label="â5")
	plot!(â[6,1:50], marker=".", label="â6")
	plot!(â[7,1:50], marker=".", label="â7")
	
end

# ╔═╡ 71109254-6d3d-11eb-2e41-71bf5c1543f9
md"
**What do you observe as `r` increases ?`**

From the time series plot, we can see that the amplitude decreases when r increases which makes sense since it is the first PCA modes that let us capture the largest information and with the increase of r, it fades away.

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
	ω̂ = Ψ * â	

# ╔═╡ 986d2f20-84ed-11eb-094d-bd6f7949b9c8
	ω = Ψ * a	

# ╔═╡ 44eb4630-84ec-11eb-023b-05aeeda59e20
begin 
	som = 0
	for i=1:n
		norm = 0
		for j=1:m
			norm += (ω̂[j,i] - ω[j,i])^2
		end
		som += norm	
	end
	Err = som / n
end

# ╔═╡ 1679e8ca-7a6e-11eb-2ca3-418d0ca74765
md"
Based on your results, how would choose $r$ such that it best balances the reconstruction accuracy and the number of required sensors ?
Justify.
"

# ╔═╡ 167978e0-7a6e-11eb-334a-5b057dc68025
md"t
**Answer :**

The choice of r will strongly depend on the error percentage we want to get. In our case, we were able to see that 7 sensors were enough to capture 99% of the data but we still get a huge error (~2100%) which is probably due to the turbulence and the sensors locations that seem quite random. 

A good trade off would be r=24 which gives less than 1% error.

"

# ╔═╡ 163c4aba-7a6e-11eb-14c7-835023e4773a
md"
The methodology presented here is quite general and can be applicable to large variety of different engineering fields.
Suggest three applications of this methodology to other problems and explain why you think if would be a good way to tackle the problem.
"

# ╔═╡ ff9de946-7a6e-11eb-3360-0f66704f508a
md"
**Answer :**

- Medical use : for example showing the correlation of Cholesterol and low density lipo-protein.

-  Finance use : financial time series, dynamic trading strategies, financial risk computations and even estimation of future stock price values.

- Face recognition.

"

# ╔═╡ ff8b5d8c-7a6e-11eb-3f98-51f763ba3743


# ╔═╡ ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5


# ╔═╡ ff55d27a-7a6e-11eb-2a81-5feaf3b230c6


# ╔═╡ 7293be7a-6d4d-11eb-2342-21bc363f601f


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


# ╔═╡ 52b0a81e-80e4-11eb-22c3-6993b0fd4d45
plot_flow_field(mesh, x̄)

# ╔═╡ f4094c04-6d39-11eb-1930-457bdb436549
# --> Plot PCA 1.

	plot_flow_field(mesh, U[:, 1])

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

	plot_flow_field(mesh, U[:, 1], p)

# ╔═╡ Cell order:
# ╠═ae0cca20-7a9d-11eb-1023-457ff1070586
# ╠═1d0a34e6-6d29-11eb-1d64-33fa134b4560
# ╠═bfc84fdc-6d25-11eb-1d13-a909ac45e366
# ╠═1b3a546e-6d29-11eb-04fe-0f4201be3100
# ╠═0801bd0a-6d2a-11eb-0071-fdd4acaf308f
# ╟─d19dff42-6d4e-11eb-19ec-ad9425b3d4c1
# ╠═6a1ee816-6d2a-11eb-026c-890add109c54
# ╟─6a05539a-6d2a-11eb-1772-a76e67e6304e
# ╠═e533f59e-6d2a-11eb-2fde-718f3c5041bd
# ╠═f808cd32-80e3-11eb-298b-750b651bf52f
# ╠═52b0a81e-80e4-11eb-22c3-6993b0fd4d45
# ╠═d9c8eb26-6d37-11eb-2718-472a8ec69bd1
# ╠═6e724220-80e5-11eb-2143-25ab4008abd8
# ╟─d52ccb44-6d39-11eb-3dff-af3ddb563f11
# ╠═d5135362-6d39-11eb-1303-1b30121d1717
# ╠═007fecfe-8525-11eb-1def-452921d156db
# ╠═16e71820-8525-11eb-2d55-3bf6bdeb2c5b
# ╠═4f4aec50-8520-11eb-1152-495ece40e16b
# ╟─be077240-85a1-11eb-1a97-678b564c16fb
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
# ╠═745985c0-851a-11eb-21ca-df99ff11c72d
# ╠═7db635dc-6d3c-11eb-2c41-218475c90ffe
# ╟─ffe4b9d0-85a5-11eb-12b6-ebcd30ecf3be
# ╟─9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
# ╠═353766a4-6d3d-11eb-2aa6-f3683b7d2d19
# ╠═34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# ╠═c77592e0-8519-11eb-0b1d-7974089a12b6
# ╠═6be4c214-6d3d-11eb-1113-598d240bc3c0
# ╟─71109254-6d3d-11eb-2e41-71bf5c1543f9
# ╟─16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
# ╠═16825226-7a6e-11eb-23e3-0baf0bad7292
# ╠═986d2f20-84ed-11eb-094d-bd6f7949b9c8
# ╠═44eb4630-84ec-11eb-023b-05aeeda59e20
# ╟─1679e8ca-7a6e-11eb-2ca3-418d0ca74765
# ╠═167978e0-7a6e-11eb-334a-5b057dc68025
# ╟─163c4aba-7a6e-11eb-14c7-835023e4773a
# ╠═ff9de946-7a6e-11eb-3360-0f66704f508a
# ╟─ff8b5d8c-7a6e-11eb-3f98-51f763ba3743
# ╟─ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5
# ╟─ff55d27a-7a6e-11eb-2a81-5feaf3b230c6
# ╟─7293be7a-6d4d-11eb-2342-21bc363f601f
# ╠═727ad7ca-6d4d-11eb-15c5-bf6a4989e458
# ╠═72675b82-6d4d-11eb-352e-8b1d5824f865
# ╠═724d6cfe-6d4d-11eb-0596-b9c552a25809
