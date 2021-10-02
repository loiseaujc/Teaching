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

# ╔═╡ b5f2e950-84d8-11eb-118b-edb60e843a6d
md"
SOME EXPLANATION BEFOR STARTING THE PROJECT.

1)We have a data matrix with dimention of 20769by400 and the each row correspond to a state variable(condition)and each column correspond to one snapshot.

2)We want to reduce the number of condition in this data matrix and say that this condition (state variable) are the most important variable.

3)When we compute the eigen value and eigen vector then we have to reorder the eigen values from the largest to the smallest (based the course info).
"

# ╔═╡ 1b3a546e-6d29-11eb-04fe-0f4201be3100
md"Dataset to be loaded : $@bind dset TextField()"

# ╔═╡ 0801bd0a-6d2a-11eb-0071-fdd4acaf308f
begin
	# --> Load the dataset.
	data = matread(dset)
	#println(data)
	
	# --> Extract the mesh and the snapshots collection.
	mesh, X = [data["x"][:], data["y"][:]], data["data"]
	size(mesh)
end;

# ╔═╡ e533f59e-6d2a-11eb-2fde-718f3c5041bd
# --> Compute the mean column.[      ] ->[] 
mean_row = mean(X,  dims = 2)

# ╔═╡ 78982890-7dc1-11eb-2a7e-9b6d7dce3d16
#make 400 copy of mean_row to make a matrix of 20769*400  dimenstion.
mean_matrix = (mean_row*ones(1,400))

# ╔═╡ d9c8eb26-6d37-11eb-2718-472a8ec69bd1
begin
	# --> Subtract mean_matrix from the data matrix.
	subtract=X-mean_matrix
	# --> Compute the covariance matrix.
	covariance =((1/399)*(subtract' * subtract))
	# --> Plot the covariance matrix.
	heatmap(covariance)
end

# ╔═╡ d5135362-6d39-11eb-1303-1b30121d1717
begin
	# --> Compute the eigendecomposition of Σ.
	eig_val,eig_vec = eigen(covariance)
	# --> Reorder the Eigen_value and Eigen vector from the largest to the smallest.
	eig_val , eig_vec= eig_val[end:-1:1] , eig_vec[:,end:-1:1]
 end


# ╔═╡ afc3b040-8411-11eb-031d-edf1a27ec9a8
# --> Plot the distribution of eigenvalues.
plot(cumsum(eig_val)./sum(eig_val))

# ╔═╡ 46adf040-8559-11eb-1e90-435dc640858b
#To findout how many row we need to have more than 99 percent variance I just tried 
#to increase the 8 from 5 and when we reach 8 you will see that we pass 99%.
sum(eig_val[1:8])./sum(eig_val)

# ╔═╡ 3ac4d7b0-84e8-11eb-05b9-e5c7d88743cd
Dia_eig_val = Diagonal(eig_val)

# ╔═╡ b1fec0c0-8551-11eb-322a-b7f4a637abbb
begin
Dia_abstract = Dia_eig_val[1:8,1:8]
# 12 is chosen based the figure of distribution
Dia_sqrt = sqrt.(Dia_abstract)
Dia_inv_sqrt = Matrix(I,8,8)/(Dia_sqrt)
end

# ╔═╡ d4f6f550-6d39-11eb-1463-770f6b0a0742
# --> Compute the PCA modes.
#by PCA_12 I mean 12 first principle component.
PCA = (1/sqrt(399))* ((subtract[:,1:8] * eig_vec[1:8,1:8])*(Dia_inv_sqrt))


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

# ╔═╡ 4cbf0822-855e-11eb-14ab-9b3a5fe2da0f
md"
You can't increase the r more than 8 because I defined PCA just with 8 row.

If you want to increase more please change the number 8 up to 20 every where in this note book.
"

# ╔═╡ 712547fe-6d3c-11eb-3b53-1982fe7051ec
# --> Truncated PCA basis Ψ (\Psi <TAB> to have Ψ displayed as variable name).
Ψ = PCA[:,1:r]
# r=8 -> 99% of variance.

# ╔═╡ 7dce6f62-6d3c-11eb-061d-654eb12763b5
# --> Compute the QR decomposition with pivot.
Q,R,P = qr(transpose(Ψ),Val(true))

# ╔═╡ cb9287d0-855e-11eb-18b7-e1b68bf65aa4
md"

As mentioned before This flow pattern is known as the Bénard-von Kàrmàn vortex street.

What I feel about this kind of sensor placement: 

1) this flow pattern can be predicted in my opinion. if you consider this flow like a signal with variable amplitude and variable wavelength then maybe we can predict the second,third and ... amplitude and also the wavelength which increase might have a specific pattern in it's increment so for this reason we just need to place the sensors inside the region of the first wave.

but I was thinking because the region close the the cylinder is unsteady and we can't predict this region we might need to place more sensors.
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
	y = X[1:r,:]
	#O = Ψ[r,:]
	# --> Build the measurement matrix Θ (\Theta <TAB>)
	#θ = Ψ[r,:]
	# --> Obtain the true low-dimensional projection (a = transpose(Ψ) * X).
	a = transpose(Ψ)*X
	
end

# ╔═╡ 34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# --> Solve Θ * â = y.\
begin
qr(a)\y
end

# ╔═╡ 6be4c214-6d3d-11eb-1113-598d240bc3c0
begin
	# --> Plot the time series.
end

# ╔═╡ 71109254-6d3d-11eb-2e41-71bf5c1543f9
md"
**What do you observe as `r` increases ?`**
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
**Answer :**
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
	plot(
		plot_flow_field(mesh, X[:, rand(1:400)]),
		plot_flow_field(mesh, X[:, rand(1:400)]),
		plot_flow_field(mesh, X[:, rand(1:400)]),
		plot_flow_field(mesh, X[:, rand(1:400)]),
		layout=(2, 2),
		size=(512, 256)
		)
end


# ╔═╡ e317aac0-8419-11eb-392e-a1aa40f6de53
#plot_flow_field(mesh, mean_row)
plot_flow_field(mesh,mean_row)

# ╔═╡ 9bf2c640-8557-11eb-3bd4-395bb1535a11
begin
	plot(
		plot_flow_field(mesh, PCA[:, 1]),
		plot_flow_field(mesh, PCA[:, 2]),
		plot_flow_field(mesh, PCA[:, 3]),
		plot_flow_field(mesh, PCA[:, 4]),
		layout=(2, 2),
		size=(512, 256)
		)
end

# ╔═╡ 1c393a10-855c-11eb-1363-e323db2ef904
plot_flow_field(mesh, PCA[:, 1], P[1:r])

# ╔═╡ 8d9f4d30-8411-11eb-1d55-3d5631167acb


# ╔═╡ f4f2dd40-855a-11eb-2520-854ec6563a6b


# ╔═╡ Cell order:
# ╠═1d0a34e6-6d29-11eb-1d64-33fa134b4560
# ╟─bfc84fdc-6d25-11eb-1d13-a909ac45e366
# ╠═b5f2e950-84d8-11eb-118b-edb60e843a6d
# ╟─1b3a546e-6d29-11eb-04fe-0f4201be3100
# ╠═0801bd0a-6d2a-11eb-0071-fdd4acaf308f
# ╠═6a1ee816-6d2a-11eb-026c-890add109c54
# ╠═e533f59e-6d2a-11eb-2fde-718f3c5041bd
# ╠═e317aac0-8419-11eb-392e-a1aa40f6de53
# ╠═78982890-7dc1-11eb-2a7e-9b6d7dce3d16
# ╠═d9c8eb26-6d37-11eb-2718-472a8ec69bd1
# ╠═d5135362-6d39-11eb-1303-1b30121d1717
# ╠═afc3b040-8411-11eb-031d-edf1a27ec9a8
# ╠═46adf040-8559-11eb-1e90-435dc640858b
# ╠═3ac4d7b0-84e8-11eb-05b9-e5c7d88743cd
# ╠═b1fec0c0-8551-11eb-322a-b7f4a637abbb
# ╠═d4f6f550-6d39-11eb-1463-770f6b0a0742
# ╠═9bf2c640-8557-11eb-3bd4-395bb1535a11
# ╟─084a5980-6d3a-11eb-0c96-7d2cf3cec629
# ╟─d7cfb6c4-6d3c-11eb-0e83-e744df9b16c5
# ╠═4cbf0822-855e-11eb-14ab-9b3a5fe2da0f
# ╠═712547fe-6d3c-11eb-3b53-1982fe7051ec
# ╠═7dce6f62-6d3c-11eb-061d-654eb12763b5
# ╠═1c393a10-855c-11eb-1363-e323db2ef904
# ╠═cb9287d0-855e-11eb-18b7-e1b68bf65aa4
# ╟─9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
# ╠═353766a4-6d3d-11eb-2aa6-f3683b7d2d19
# ╠═34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# ╠═6be4c214-6d3d-11eb-1113-598d240bc3c0
# ╠═71109254-6d3d-11eb-2e41-71bf5c1543f9
# ╟─16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
# ╠═16825226-7a6e-11eb-23e3-0baf0bad7292
# ╟─1679e8ca-7a6e-11eb-2ca3-418d0ca74765
# ╠═167978e0-7a6e-11eb-334a-5b057dc68025
# ╟─163c4aba-7a6e-11eb-14c7-835023e4773a
# ╠═ff9de946-7a6e-11eb-3360-0f66704f508a
# ╟─ff8b5d8c-7a6e-11eb-3f98-51f763ba3743
# ╟─ff6fe7fa-7a6e-11eb-31f1-8fcbaf7aa9a5
# ╟─ff55d27a-7a6e-11eb-2a81-5feaf3b230c6
# ╟─7293be7a-6d4d-11eb-2342-21bc363f601f
# ╟─727ad7ca-6d4d-11eb-15c5-bf6a4989e458
# ╟─72675b82-6d4d-11eb-352e-8b1d5824f865
# ╟─724d6cfe-6d4d-11eb-0596-b9c552a25809
# ╠═8d9f4d30-8411-11eb-1d55-3d5631167acb
# ╠═f4f2dd40-855a-11eb-2520-854ec6563a6b
