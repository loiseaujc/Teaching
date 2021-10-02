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
	data = matread("cylinder_dataset.mat")
	
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

# ╔═╡ d52ccb44-6d39-11eb-3dff-af3ddb563f11
md"
 **Interpration :**
"#From the heatmap we can observe a very structured and repetitive structure. Therefore we can infer that our system has some sort of periodic behaviour and tha it is possible to reconstruct it in an accurate manner using only a few PCA modes.

# ╔═╡ 05028eee-6d3a-11eb-3fbd-a75908821bb1
md"
**Pattern :**
"#According to our results, we would need at least 8 PCA modes to capture 99% of the Kinetic energy.
#From my very basic understanding of Fluid mechanics, I believe these snapshots to be the behaviour of the vortexes formed when the fluid colides with the object over time. This would explain why with this 8 snapshots we can reconstruct the whole model by making linear combinations of them, since the model has a periodic behaviour that repeats itself each cycle. Furthermore, the system presents a horizontal symetric behaviour, hence it is likely that we would only need to follow the behaviour of a vortex in the top or bottom and we could predict the behaviour of its counterpart. 

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
md"Rank `r` of the PCA basis : $@bind r Slider(1:100, default=2, show_value=true)"

# ╔═╡ 4c361850-857a-11eb-3275-9f53013790f2
#Interpretation of sensor locations.
#As mentioned before, the system shows a periodic behaviour which repeats over time and a horizontal symetric component. Also we can observe that as the vortexes get further away from the cylinder, their behaviour is still the same but with a different "Amplitude/Magnitude". Taking this into account, it would make sense that the majority of the sensors are placed in the region where the first Vortex forms, with most of them along the center line, where interactions between the top and bottom vortexes occur, and with a few extra sensors at either the top or bottom halve, just to accurately reconstruct the behaviour of one vortex and predict that of its counter part. 

# ╔═╡ 9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
md"
Denoting by `a` the projection `tranpose(Ψ) * X` (i.e. the projection of the data into the leading PCA subspace), `y = X[p, :]` the measurements and `Θ = Ψ[p, :]` the measurement matrix, use the least-squares technique to solve

```math
 \mathbf{y} = \boldsymbol{\Theta} \hat{\mathbf{a}}
```

and compare the the time-series of the estimated `â` coefficients with the true ones in `a` for varying values of `r`.
What do you observe ?

"

# ╔═╡ 71109254-6d3d-11eb-2e41-71bf5c1543f9
md"
**What do you observe as `r` increases ?`**
" #As we increase the value of r the model is able to represent more accurately the behaviour of our system since there is more information available.

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

# ╔═╡ 1679e8ca-7a6e-11eb-2ca3-418d0ca74765
md"
Based on your results, how would choose $r$ such that it best balances the reconstruction accuracy and the number of required sensors ?
Justify.
" 

# ╔═╡ 167978e0-7a6e-11eb-334a-5b057dc68025
md"
**Answer :**
"#I would use a function like the one shown above (Example of balancing function). Where we would set a limit at the desired accuracy and calculate exactly how many sensors we would need to get that level of precision.

# ╔═╡ 163c4aba-7a6e-11eb-14c7-835023e4773a
md"
The methodology presented here is quite general and can be applicable to large variety of different engineering fields.
Suggest three applications of this methodology to other problems and explain why you think if would be a good way to tackle the problem.
"

# ╔═╡ ff9de946-7a6e-11eb-3360-0f66704f508a
md"
**Answer :**
"#As seen in class there are multiple applications for this methodology such as Sensor positioning for Climate Prediction/Forcasting, Face Recognition/Reconstruction, and in Quality control of mechanical properties in a machine.  
#In Climate applications it can be used to define the number and location of sensors necessary to create a reliable model used to predict changes in temperature, humidity, wind, etc.
#For face recognition software it allows us to define the areas of the face where the computer program should focus its attention in order to be able to recognize a face efficiently and also be able to differentiate it form others.
#For Quality control based on previous experiments performed in a model of a specific machine, we could use this method to place our sensors in the areas where the biggest stresses or deformations tend to be be presented, and use it as a quality check to make sure the machine is able to perform as desired.

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


# ╔═╡ e533f59e-6d2a-11eb-2fde-718f3c5041bd
begin
# --> Compute the mean flow.
	m,n = size(X);
	#Xs = sum(X, dims=2)
	#Xm = Xs/n
	Xm = mean(X, dims=2)
	plot_flow_field(mesh, Xm)
end

# ╔═╡ 6f171c4e-81a3-11eb-07f3-efff45d72c9d
maximum(Xm)

# ╔═╡ d9c8eb26-6d37-11eb-2718-472a8ec69bd1
begin
	# --> Subtract mean from the data matrix.
	B = X.-Xm
	
	# --> Compute the covariance matrix.
	Σ = (1/(n-1)) * (B'*B)
	# --> Plot the covariance matrix.
	heatmap(Σ, yflip=true)

end

# ╔═╡ d5135362-6d39-11eb-1303-1b30121d1717
begin
	# --> Compute the eigendecomposition of Σ.
	Λ, V = eigen(Σ) #Λ are the eigen values and V its eigenvectors
	# --> Plot the distribution of eigenvalues.
	Λ = Λ[end:-1:1]
	V = V[:,end:-1:1]
	plot(cumsum(Λ)./sum(Λ))

end

# ╔═╡ c43b7aa0-80f4-11eb-1e6b-fdf87e9f1f57
Λd = Diagonal(Λ)

# ╔═╡ eb56c540-80f4-11eb-19b5-0fa8103b9bb8
begin
	trg = 0.99999 * tr(Λd)  #Desired % of Kinetic energy to be captured
	i=1                  #Iteration counter
	val = Λ[1]           #First eigenvalue
	while val<=trg       #Checks if current modes capture the desired energy
		val = val + Λ[i+1] #If not, add energy or the next eigenvalue
		i = i+1            #Increase the iteration counter
	end
	modes = i           #Number of PCA modes to be considered
end

# ╔═╡ a5145e30-8577-11eb-10f1-ef433716665d
begin
	nmodes = modes + 80 #Adding extra modes for comparision purposes.
end

# ╔═╡ d4f6f550-6d39-11eb-1463-770f6b0a0742
# --> Compute the PCA modes.
# Truncation to 99%
U = (1/sqrt(n-1))*B*V[:,1:nmodes]*Λd[1:nmodes,1:nmodes]^-0.5

# ╔═╡ 712547fe-6d3c-11eb-3b53-1982fe7051ec
# --> Truncated PCA basis Ψ (\Psi <TAB> to have Ψ displayed as variable name).
Ψ= U[:,1:r]

# ╔═╡ 7dce6f62-6d3c-11eb-061d-654eb12763b5
# --> Compute the QR decomposition with pivot.
Q , R, ptemp = qr(Ψ',Val(true))


# ╔═╡ 66adcb60-8345-11eb-12a4-b3c072cb5429
p = ptemp[1:r]

# ╔═╡ 353766a4-6d3d-11eb-2aa6-f3683b7d2d19
begin
	# --> Takes the measurements y.
	y = X[p,:]
	# --> Build the measurement matrix Θ (\Theta <TAB>)
	Θ = Ψ[p,:]
	
	# --> Obtain the true low-dimensional projection (a = transpose(Ψ) * X).
	a = Ψ' * X

end

# ╔═╡ 34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# --> Solve Θ * â = y.
ahat = Θ\y

# ╔═╡ 6be4c214-6d3d-11eb-1113-598d240bc3c0
begin
	# --> Plot the time series.
	xaxis = 1:n
	â = transpose(ahat)
	plot(xaxis, â[:,1], marker= (:cicle,3),xlims=(0,70))
	#plot(â[:,1], marker = (:cirle,3), xlims=(0,70))

end

# ╔═╡ f9103fe0-84b9-11eb-1afd-915de7fb65f6
begin
b = a'
plot(xaxis, b[:,1], marker= (:cicle,3),xlims=(0,70))
	#plot(b[:,1], marker = (:circle,3), xlims=(0,70))
end

# ╔═╡ 9a343282-84f8-11eb-1ebf-8d132cb9f926
begin		
	w_hat = Ψ * ahat
	w = Ψ * a
	N2 = 0
	for j=1:n
		N2= N2 + (norm(w_hat[:,j]-w[:,j]))^2 #Sum of the squared distances.
	end
	Err = (N2/n)*100 #Computes the error.
end

# ╔═╡ 2f29c260-8517-11eb-3551-b5b78c838be4
#Example of balancing function.
begin
	E = 100
	jj = 1
	r_real = 0 #Starts the count of the number of sensors required.
	tol = 1 #Desired reconstruction accuracy
	while E >= tol
		f_hat = Ψ[:,1:jj] * ahat[1:jj,:]
		f = Ψ[:,1:jj] * a[1:jj,:]
		N3 = 0	
			for j = 1:n
				N3 = N3 + (norm(f_hat[:,j]-w[:,j]))^2
			end
		jj = jj + 1
		r_real = r_real + 1 #Adds 1 to the  count at every loop of the while cycle.
		E = (N3/n)*100  #Computes the error at every loop.
	end
	r_real
end

# ╔═╡ 0f11934e-8575-11eb-19f2-f3c9e35eea89
p[1:r_real]

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

# ╔═╡ ca8f46c0-817b-11eb-30a0-8f8c536fad58
# --> Plot PCA 5.
plot_flow_field(mesh, U[:, 5])

# ╔═╡ dafe1fe0-817b-11eb-0a90-3b4763277556
# --> Plot PCA 6.
plot_flow_field(mesh, U[:, 6])

# ╔═╡ e063ced0-817b-11eb-3009-a54b15550678
# --> Plot PCA 7.
plot_flow_field(mesh, U[:, 7])

# ╔═╡ e6166540-817b-11eb-2947-37bf51d21be1
# --> Plot PCA 8.
plot_flow_field(mesh, U[:, 8])

# ╔═╡ 7db635dc-6d3c-11eb-2c41-218475c90ffe
# --> Plot the first PCA mode superimposed with the sensor locations.
plot_flow_field(mesh, U[:, 1], p)

# ╔═╡ 79e1a170-8575-11eb-38dd-69513b8ffa7c
# --> Plot the first PCA mode superimposed with the optimal number of sensors.
plot_flow_field(mesh, U[:, 1], p[1:r_real])

# ╔═╡ Cell order:
# ╠═1d0a34e6-6d29-11eb-1d64-33fa134b4560
# ╟─bfc84fdc-6d25-11eb-1d13-a909ac45e366
# ╟─1b3a546e-6d29-11eb-04fe-0f4201be3100
# ╠═0801bd0a-6d2a-11eb-0071-fdd4acaf308f
# ╟─d19dff42-6d4e-11eb-19ec-ad9425b3d4c1
# ╟─6a1ee816-6d2a-11eb-026c-890add109c54
# ╟─6a05539a-6d2a-11eb-1772-a76e67e6304e
# ╠═e533f59e-6d2a-11eb-2fde-718f3c5041bd
# ╠═6f171c4e-81a3-11eb-07f3-efff45d72c9d
# ╠═d9c8eb26-6d37-11eb-2718-472a8ec69bd1
# ╠═d52ccb44-6d39-11eb-3dff-af3ddb563f11
# ╠═d5135362-6d39-11eb-1303-1b30121d1717
# ╠═c43b7aa0-80f4-11eb-1e6b-fdf87e9f1f57
# ╠═eb56c540-80f4-11eb-19b5-0fa8103b9bb8
# ╠═a5145e30-8577-11eb-10f1-ef433716665d
# ╠═d4f6f550-6d39-11eb-1463-770f6b0a0742
# ╠═f4094c04-6d39-11eb-1930-457bdb436549
# ╠═f7dadaa0-6d39-11eb-041b-396f59ed4ed8
# ╠═fba0f79e-6d39-11eb-1506-c7a6a40ca7ee
# ╠═fb88bc94-6d39-11eb-2591-293bff72a732
# ╠═ca8f46c0-817b-11eb-30a0-8f8c536fad58
# ╠═dafe1fe0-817b-11eb-0a90-3b4763277556
# ╠═e063ced0-817b-11eb-3009-a54b15550678
# ╠═e6166540-817b-11eb-2947-37bf51d21be1
# ╠═05028eee-6d3a-11eb-3fbd-a75908821bb1
# ╟─084a5980-6d3a-11eb-0c96-7d2cf3cec629
# ╠═d7cfb6c4-6d3c-11eb-0e83-e744df9b16c5
# ╠═712547fe-6d3c-11eb-3b53-1982fe7051ec
# ╠═7dce6f62-6d3c-11eb-061d-654eb12763b5
# ╠═66adcb60-8345-11eb-12a4-b3c072cb5429
# ╠═7db635dc-6d3c-11eb-2c41-218475c90ffe
# ╠═4c361850-857a-11eb-3275-9f53013790f2
# ╟─9b6cb7e0-6d3c-11eb-2df5-e1e2a378cc05
# ╠═353766a4-6d3d-11eb-2aa6-f3683b7d2d19
# ╠═34b3164c-6d3d-11eb-08bb-b71b4d6ad947
# ╠═6be4c214-6d3d-11eb-1113-598d240bc3c0
# ╠═f9103fe0-84b9-11eb-1afd-915de7fb65f6
# ╠═71109254-6d3d-11eb-2e41-71bf5c1543f9
# ╟─16a9a614-7a6e-11eb-2fb8-79dafc35b2bf
# ╠═9a343282-84f8-11eb-1ebf-8d132cb9f926
# ╠═2f29c260-8517-11eb-3551-b5b78c838be4
# ╠═0f11934e-8575-11eb-19f2-f3c9e35eea89
# ╠═79e1a170-8575-11eb-38dd-69513b8ffa7c
# ╠═1679e8ca-7a6e-11eb-2ca3-418d0ca74765
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
