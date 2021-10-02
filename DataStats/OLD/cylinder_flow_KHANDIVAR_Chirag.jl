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
	
	Matlab_data = matread("cylinder_dataset.mat")
	
	# --> Extract the mesh and the snapshots collection.
	
	mesh, X = [Matlab_data["x"][:], Matlab_data["y"][:]], Matlab_data["data"]

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
	
	X̄ = mean(X, dims=2)

# ╔═╡ d9c8eb26-6d37-11eb-2718-472a8ec69bd1
begin
	
	m,n = size(X)
	
	# --> Subtract mean from the data matrix.
	
	B = X.- X̄
	
	# --> Compute the covariance matrix.
	
 	Σ = (1/(n-1)) * (B'* B)
	
	# --> Plot the covariance matrix.
	
	heatmap(Σ)
	
end

# ╔═╡ d52ccb44-6d39-11eb-3dff-af3ddb563f11
md"
	Interpration of heatmap : 
° The above obtained heatmap can be looked as 400 * 400 pixels image, in which each pixel carries a value of each component of covariance matrix(Σ).

° In order to get a more cleared overview, we can plot the heatmap in terms of abssolute value of Σ as below. From this heatmap we can say that there are approximately 30-40% pixels are in black color which means they don't carry any values. 

° In other words, one can say that they don't carry vital information and are not important statistically, so we can eliminate them easily while doing Principle Component Analysis.

° Other pixels, which are in yellow & orange color are most important in terms of statistics because they carry the required major information and in fact, from them we can select some as our principle components.
"

# ╔═╡ 00bf80e0-858f-11eb-0660-234b9783a2a2
heatmap(abs.(Σ))

# ╔═╡ d5135362-6d39-11eb-1303-1b30121d1717
begin
	
	# --> Compute the eigendecomposition of Σ.
	
	λ, V = eigen(Σ)
	
	λ_ = sort!(λ, rev = true)
	#V_ = view(V, :, size(V)[2]:-1:1)
	V_ = V[:, end:-1:1]
	
	
	# --> Plot the distribution of eigenvalues.
	
	plot(cumsum(λ_)/sum(λ_))

end

# ╔═╡ d4f6f550-6d39-11eb-1463-770f6b0a0742
begin
	
	λ̂ = Diagonal(λ_) 
	
	λₚ = λ̂[1:10,1:10]
	
	# --> Compute the PCA modes.
	
	λₑ = Matrix(I,10,10) / λₚ^(1/2)
	
	U = 1/sqrt(n-1) * B * V_[:,1:10] * λₑ
end

# ╔═╡ b0053c0e-8590-11eb-1ce7-377b92e3a3a5
λₚ

# ╔═╡ a8253e50-8590-11eb-351b-2112fbe0452a
md"
	Given that each eigenvalue is directly related to the amount of turbulent kinetic     energy captured by the corresponding PCA mode; after looking at their values, in 		my opinion I only need 10 modes to keep in my analysis to capture 99% of the 		 kinetic energy.
"

# ╔═╡ 05028eee-6d3a-11eb-3fbd-a75908821bb1
md"
	Pattern : 

	By looking at the patterns of four principle components and comparing them with
	the random snapshots from the database, I can derive that the first principle   		component reprsents obviously (according to definition of PCA) the closest looking
	image of original flow and gives the most imformation among those four modes.

	After that when we look at 2nd one and compare it to 1st one, it looks noisier  		than the 1st one which means it carries less vital info than previous one.

	Similarly as we analyse the 3rd principle component, the vortices become more   		complex, which stastically means that it is carrying less info than previous two 	 and also that info is not so vital as those 2 have.

	And it gets even more complex for 4th component.
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
begin
	
	# --> Compute the QR decomposition with pivot.

	Q, R, pivot = qr(transpose(Ψ), Val(true))
	p = pivot[1:r]
	
end

# ╔═╡ 414e6f20-8596-11eb-02b0-ffe620c5f4a7
md"

	By using physical intution, these sensor locations does not make sense to me because they are placed on the other side of the flow, of which we want to measure the       velocity. The chances of error are higher there becase of less availibilty of vital  data.
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
	a = transpose(Ψ) * X

end

# ╔═╡ 34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# --> Solve Θ * â = y.
	
	â = Θ\y

# ╔═╡ 6be4c214-6d3d-11eb-1113-598d240bc3c0
begin
	# --> Plot the time series.
	x̂ = 1:n
	plot(x̂,â')
end

# ╔═╡ cc384aa0-8593-11eb-0911-ed92a71a6d4e
plot(x̂,a')

# ╔═╡ 71109254-6d3d-11eb-2e41-71bf5c1543f9
md"
	Comparing the the time-series of the estimated â coefficients with the true ones in a for varying values of r, I can observe that estimated â coefficients try to come     closer to the actual ones in a as the value of r increases but though they are still very far from actual ones in terms of amplitude.
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
begin
	
	ω̂ = Ψ * â
	ω = Ψ * a
	Σ̄ = norm(ω̂ - ω)^2
	ϵ = (1/n) * Σ̄
	
end

# ╔═╡ 1679e8ca-7a6e-11eb-2ca3-418d0ca74765
md"
Based on your results, how would choose $r$ such that it best balances the reconstruction accuracy and the number of required sensors ?
Justify.
"

# ╔═╡ 167978e0-7a6e-11eb-334a-5b057dc68025
md"
**Answer :**
"

# ╔═╡ 163c4aba-7a6e-11eb-14c7-835023e4773a
md"
The methodology presented here is quite general and can be applicable to large variety of different engineering fields.
Suggest three applications of this methodology to other problems and explain why you think if would be a good way to tackle the problem.
"

# ╔═╡ ff9de946-7a6e-11eb-3360-0f66704f508a
md"
	This methodology is used in water quality management. This method can be used to      assess water quality at different temporal and spatial levels for groundwater         classification. Total dissolved solids, total hardness, chloride, fluoride,          nitrate (N) can be identified as important variables affecting water classification 	and quality.

courtesy : Evaluation of hierarchically weighted principal component analysis for
water quality management at Jiaozuo mine by Xuan Guoa, XiaoXin Zhangc, HuanChuang Yueb

	The use of principle component analysis can sensitively detect the change in the 		pattern of the frequency spectrums from different bearing fault .

courtesy : Clustering of frequency spectrums from different bearing fault using 					principle component analysis by M.F.M Yusof, C.K.E Nizwan, S.A Ong, 			and M. Q. M Ridzuan
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


# ╔═╡ 7c2a5730-84ec-11eb-18f6-f3be44dca4b3
plot_flow_field(mesh, X̄)

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
# ╠═1d0a34e6-6d29-11eb-1d64-33fa134b4560
# ╟─bfc84fdc-6d25-11eb-1d13-a909ac45e366
# ╟─1b3a546e-6d29-11eb-04fe-0f4201be3100
# ╠═0801bd0a-6d2a-11eb-0071-fdd4acaf308f
# ╟─d19dff42-6d4e-11eb-19ec-ad9425b3d4c1
# ╟─6a1ee816-6d2a-11eb-026c-890add109c54
# ╟─6a05539a-6d2a-11eb-1772-a76e67e6304e
# ╠═e533f59e-6d2a-11eb-2fde-718f3c5041bd
# ╠═7c2a5730-84ec-11eb-18f6-f3be44dca4b3
# ╠═d9c8eb26-6d37-11eb-2718-472a8ec69bd1
# ╟─d52ccb44-6d39-11eb-3dff-af3ddb563f11
# ╠═00bf80e0-858f-11eb-0660-234b9783a2a2
# ╠═d5135362-6d39-11eb-1303-1b30121d1717
# ╠═d4f6f550-6d39-11eb-1463-770f6b0a0742
# ╠═b0053c0e-8590-11eb-1ce7-377b92e3a3a5
# ╟─a8253e50-8590-11eb-351b-2112fbe0452a
# ╠═f4094c04-6d39-11eb-1930-457bdb436549
# ╠═f7dadaa0-6d39-11eb-041b-396f59ed4ed8
# ╠═fba0f79e-6d39-11eb-1506-c7a6a40ca7ee
# ╠═fb88bc94-6d39-11eb-2591-293bff72a732
# ╟─05028eee-6d3a-11eb-3fbd-a75908821bb1
# ╟─084a5980-6d3a-11eb-0c96-7d2cf3cec629
# ╟─d7cfb6c4-6d3c-11eb-0e83-e744df9b16c5
# ╠═712547fe-6d3c-11eb-3b53-1982fe7051ec
# ╠═7dce6f62-6d3c-11eb-061d-654eb12763b5
# ╠═7db635dc-6d3c-11eb-2c41-218475c90ffe
# ╟─414e6f20-8596-11eb-02b0-ffe620c5f4a7
# ╟─9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
# ╠═353766a4-6d3d-11eb-2aa6-f3683b7d2d19
# ╠═34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# ╠═6be4c214-6d3d-11eb-1113-598d240bc3c0
# ╠═cc384aa0-8593-11eb-0911-ed92a71a6d4e
# ╟─71109254-6d3d-11eb-2e41-71bf5c1543f9
# ╟─16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
# ╠═16825226-7a6e-11eb-23e3-0baf0bad7292
# ╟─1679e8ca-7a6e-11eb-2ca3-418d0ca74765
# ╠═167978e0-7a6e-11eb-334a-5b057dc68025
# ╟─163c4aba-7a6e-11eb-14c7-835023e4773a
# ╟─ff9de946-7a6e-11eb-3360-0f66704f508a
# ╟─ff8b5d8c-7a6e-11eb-3f98-51f763ba3743
# ╟─ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5
# ╟─ff55d27a-7a6e-11eb-2a81-5feaf3b230c6
# ╟─7293be7a-6d4d-11eb-2342-21bc363f601f
# ╟─727ad7ca-6d4d-11eb-15c5-bf6a4989e458
# ╟─72675b82-6d4d-11eb-352e-8b1d5824f865
# ╟─724d6cfe-6d4d-11eb-0596-b9c552a25809
