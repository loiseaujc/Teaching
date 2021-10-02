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
4. Compute and plot the first four PCA modes using `plot_flow_field(mesh, U[:, i])`. Even though you may not be familiar with fluid dynamics, try to explain the patterns you observe.
"

# ╔═╡ abe26880-7cec-11eb-2ef6-11af291624fc
m,n = size(X)

# ╔═╡ e533f59e-6d2a-11eb-2fde-718f3c5041bd
# --> Compute the mean flow.
X̃ = mean(X,dims=2)

# ╔═╡ c2bef5e0-817a-11eb-0abd-a3f738660b09
size(X̃)

# ╔═╡ d9c8eb26-6d37-11eb-2718-472a8ec69bd1
begin
	# --> Subtract mean from the data matrix.
	#
	#for i=1:400
		#X[:,i] .-= X̃
	#end
	 B = X.-X̃
	#plot_flow_field(mesh, X̃)
	# --> Compute the covariance matrix.
	Σ=(1/(n-1))* (B'*B)
	# --> Plot the covariance matrix.
	heatmap(Σ)
end

# ╔═╡ d52ccb44-6d39-11eb-3dff-af3ddb563f11
md"
 **Interpration :**
The covariance matrix is a measure of the joint variability of two variables
in a dataset. In other words, it lets us know how any two features vary from each other. The properties of this covariance matrix is that it is symmetric, positive and semi definite.

The main idea behind covariance matrix is that it can classify three types of relationships:

1). Relationship with positive trends.

2). Relationship with negative trends.

3). No relationship because there is no trend.

And from the heatmap plot above we can see that the heat streak have a positive slope which kind of tell us there is a positive trend relationship among the variables in our data!

"

# ╔═╡ d5135362-6d39-11eb-1303-1b30121d1717
begin
	# --> Compute the eigendecomposition of Σ.
	Λ, V = eigen(Σ)
	#Λ = Λ[:,end:-1:1]
	Λ , V = Λ[end:-1:1],V[:, end:-1:1] #this line orders the eigen values and vectors into decreasing order respectively #
	
	# --> Plot the distribution of eigenvalues.
	
	#xmin=-0.0001
	#xmax=0.0001
	#ymin=
	#ymax=0.01
	#ylim=(ymin,ymax)
	#cumsum(Λ)./sum(Λ)
	plot(cumsum(Λ)./sum(Λ))
end

# ╔═╡ 7910e8b0-80fb-11eb-2353-4338d6b4ecb2
#=
To calculate the number of PCA modes required to capture 99% of the Kinetic energy for the purpose of our analysis. I tried to compute the sum of the required number of largest eigen values divided by the total sum of all the eigen values. 

And from the results we can see that we can already capture 90% of the kinetic energy with the first 4 eigen values but to capture 99% we need to either take the first 8 eigen values (PCA modes).
=#
begin
	nmodes=8
	pervar=ones(8,1)
	for i=1:nmodes
	pervar[i,:] = [sum(Λ[1:i])./sum(Λ)]*100
	end
	pervar
end

# ╔═╡ b4cb1a10-8286-11eb-17b0-ede3e8ea70ea
Λdiag = Diagonal(Λ)

# ╔═╡ 2a08229e-7ec7-11eb-1add-9968ae110319
begin
	Usvd, Sig, W= svd(Σ)
end

## I computed this SVD just as a sort of way to confirm the values I had are correct and that the eigen decomposition worked as expected.

# ╔═╡ 5bffe3d0-8285-11eb-27c3-d9fddfef3bca
Λc = Λdiag[1:50,1:50]

# The first 14 values from the diagonalised Λ values were chosen for the purpose of further calculations

# ╔═╡ d4f6f550-6d39-11eb-1463-770f6b0a0742
# --> Compute the PCA modes.
#= 
According to the formula given, Λrt was calculated as a way to make the final equation more readable and avoid some unnecessary erors.

An equivalent number of V eigen vectors corresponding to the number of eigen values choosen was taken.
=#

begin
	Λrt = Matrix(I,50,50) * Λc ^ -(1/2)
	U = 1/sqrt(n-1)*B*V[:,1:50]*Λrt
end

# ╔═╡ 1b1a31f0-802c-11eb-39a6-dbe07fa0661b
#Projection of the Snapshots
begin
	D=transpose(U)*B
end

# ╔═╡ 05028eee-6d3a-11eb-3fbd-a75908821bb1
md"
**Pattern :**

From the plots of the first four PCA modes, I can observe that for the first and second modes, we get a good idea of the flow around the cylinder looks like similar to a smooth flow happening in chunks.

For PCA3 and PCA4, their plot look similar although there are differences. And we can notice for both PCA3 and PCA4 that close to the cylinder, we have some more intricate details in the plot as some type of flow patterns begin to emerge. 
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
begin
	_, _, P = qr(transpose(Ψ), Val(true))
	P = P[1:r]
end

# ╔═╡ 7eac41c0-84d9-11eb-2034-7d641df037c9
md"
**Do the sensor placements make sense?**

For this plot, I do not totally agree with the placement of the sensors, the main reason is that there is no evidence (as far as I know) that positions chosen by this algorithm will help us best capture the flow. Using physical intuition I would have thought that the sensors would be placed not as far from the cylinder and evenly spread out.
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
	y = X[P , :]
	# --> Build the measurement matrix Θ (\Theta <TAB>)
	Θ = Ψ[P , :]
	# --> Obtain the true low-dimensional projection (a = transpose(Ψ) * X).
	a = transpose(Ψ)*X
end

# ╔═╡ 34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# --> Solve Θ * â = y.
#solve using least squares method
â = Θ\y

# ╔═╡ 47982300-8576-11eb-0eb6-490f6ba54f90
#################################################################
##	Here is a  time series plot comparing the estimated     #####
##	   â coefficients a with the true ones a			     #####
##															#####
#################################################################

# ╔═╡ 6be4c214-6d3d-11eb-1113-598d240bc3c0
begin
	# --> Plot the time series.
	xaxis = 1:n
	#a_trs = transpose ahat
	plot(xaxis, [â[1,:], a[1,:]], xlims=(0,200), label = ["estimated â" "true a"])
end

# ╔═╡ 71109254-6d3d-11eb-2e41-71bf5c1543f9
md"
**What do you observe as `r` increases ?`**

As you increase the value of r, the amplitude of the estimated coefficients â tend to overshoot those of the true a. For r = 1, 2 the amplitudes of both coefficients are almost equal (almost in sync at r=1) with the shape of the estimated â having some deform for r = 2. As we further increase r, I observed that they (â, a) start to sort of converge.
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
	Ω = ω̂ - ω
	sum_of_norm = (norm(Ω,2))^2;
	error = 1/n * sum_of_norm 
	PercentError = error*100
end

# ╔═╡ 1679e8ca-7a6e-11eb-2ca3-418d0ca74765
md"
Based on your results, how would choose $r$ such that it best balances the reconstruction accuracy and the number of required sensors ?
Justify.
"

# ╔═╡ 167978e0-7a6e-11eb-334a-5b057dc68025
md"
**Answer :**

We would like to choose r such that we have a good reconstruction accuracy for the least number of sensors. When we increase r (rank), we increase the number of linearly idependent columns and rows.

I would choose the value of r that gives us a 99% accuracy for our reconstruction. And from the results of the errors above, at r = 23 we have an error of 1.2%. This means with 23 sensors we can accurately reconstruct the whole vorticity field.
"

# ╔═╡ 163c4aba-7a6e-11eb-14c7-835023e4773a
md"
The methodology presented here is quite general and can be applicable to large variety of different engineering fields.
Suggest three applications of this methodology to other problems and explain why you think if would be a good way to tackle the problem.
"

# ╔═╡ ff9de946-7a6e-11eb-3360-0f66704f508a
md"
**Answer :**

This methodology presented in this workbook can be applied to other problems including 

1). Medical field : Predicting the chances of a person of getting a certain disease or not. We can use PCA to identify the major genetic markers in a large dataset that can help to determine whether a person is susceptible or not to the disease with a high degree of accuracy

2). Anomaly and Fraud Detection : Anomaly detection relies on reconstruction error. To analyse Credit card transactions to detect fraudulence, PCA can be performed on the original dataset and the right number Principal components chosen for the recosntruction. We want the reconstruction error for rare transactions—the ones that are most likely to be fraudulent—to be as high as possible and the reconstruction error for the rest to be as low as possible.

If we keep too many principal components, PCA may too easily reconstruct the original transactions, so much so that the reconstruction error will be minimal for all of the transactions. If we keep too few principal components, PCA may not be able to reconstruct any of the original transactions well enough—not even the normal, nonfraudulent transactions.

3). Detecting and Visualizing Computer Network Attacks : Distributed Denial of Service (DDoS) attacks and Network Probe (NP) attacks. Both attacks have a common characteristic of utilizing many packets as seen by the network interface. PCA can be used to reduce the dimensionality of the feature vectors to enable better visualization and analysis of the data.

The loading values of the features on the first and second principal
components to identify an attack. For normal traffic, loading values appear to be similar, while during an attack the loading values differ significantly for the first two principal components. A threshold value could be used to make such a distinction.
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


# ╔═╡ 8e896af0-7dc2-11eb-3e7d-fbc7c2b8810c
plot_flow_field(mesh, X̃)

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
plot_flow_field(mesh, U[:, 1], P)

# ╔═╡ Cell order:
# ╠═1d0a34e6-6d29-11eb-1d64-33fa134b4560
# ╟─bfc84fdc-6d25-11eb-1d13-a909ac45e366
# ╟─1b3a546e-6d29-11eb-04fe-0f4201be3100
# ╠═0801bd0a-6d2a-11eb-0071-fdd4acaf308f
# ╟─d19dff42-6d4e-11eb-19ec-ad9425b3d4c1
# ╟─6a1ee816-6d2a-11eb-026c-890add109c54
# ╟─6a05539a-6d2a-11eb-1772-a76e67e6304e
# ╠═abe26880-7cec-11eb-2ef6-11af291624fc
# ╠═e533f59e-6d2a-11eb-2fde-718f3c5041bd
# ╠═c2bef5e0-817a-11eb-0abd-a3f738660b09
# ╠═8e896af0-7dc2-11eb-3e7d-fbc7c2b8810c
# ╠═d9c8eb26-6d37-11eb-2718-472a8ec69bd1
# ╟─d52ccb44-6d39-11eb-3dff-af3ddb563f11
# ╠═d5135362-6d39-11eb-1303-1b30121d1717
# ╠═7910e8b0-80fb-11eb-2353-4338d6b4ecb2
# ╠═b4cb1a10-8286-11eb-17b0-ede3e8ea70ea
# ╠═2a08229e-7ec7-11eb-1add-9968ae110319
# ╠═5bffe3d0-8285-11eb-27c3-d9fddfef3bca
# ╠═d4f6f550-6d39-11eb-1463-770f6b0a0742
# ╠═1b1a31f0-802c-11eb-39a6-dbe07fa0661b
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
# ╟─7eac41c0-84d9-11eb-2034-7d641df037c9
# ╟─9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
# ╠═353766a4-6d3d-11eb-2aa6-f3683b7d2d19
# ╠═34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# ╠═47982300-8576-11eb-0eb6-490f6ba54f90
# ╠═6be4c214-6d3d-11eb-1113-598d240bc3c0
# ╟─71109254-6d3d-11eb-2e41-71bf5c1543f9
# ╟─16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
# ╠═16825226-7a6e-11eb-23e3-0baf0bad7292
# ╟─1679e8ca-7a6e-11eb-2ca3-418d0ca74765
# ╟─167978e0-7a6e-11eb-334a-5b057dc68025
# ╟─163c4aba-7a6e-11eb-14c7-835023e4773a
# ╟─ff9de946-7a6e-11eb-3360-0f66704f508a
# ╟─ff8b5d8c-7a6e-11eb-3f98-51f763ba3743
# ╟─ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5
# ╟─ff55d27a-7a6e-11eb-2a81-5feaf3b230c6
# ╟─7293be7a-6d4d-11eb-2342-21bc363f601f
# ╟─727ad7ca-6d4d-11eb-15c5-bf6a4989e458
# ╟─72675b82-6d4d-11eb-352e-8b1d5824f865
# ╟─724d6cfe-6d4d-11eb-0596-b9c552a25809
