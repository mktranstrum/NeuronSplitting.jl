include("Data.jl")
include("Model.jl")


module NeuronSplitting

using PyPlot

import Random: seed!
using Main.Data
using Main.Model

figure()
plot_data()


seed!(1985)
# Start with a single neuron
θ0 = randn(3*1 + 1)* 1e0
plot(xfine, fhat(xfine, θ0), label = "initial")

@show loss(θ0, λ)
θf = train(θ0)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()

score, θ0, dθ = optimal_split(θf)
@show Nh(θ0), score
τ = line_search(θ0, dθ)
@show τ
figure()
plot_data()
plot(xfine, fhat(xfine, θ0), label = "Initial")
plot(xfine, fhat(xfine, θ0 + dθ*τ), label = "Perturbed")
θf = train(θ0 + dθ*τ)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()


score, θ0, dθ = optimal_split(θf)
@show Nh(θ0), score
τ = line_search(θ0, dθ)
@show τ
figure()
plot_data()
plot(xfine, fhat(xfine, θ0), label = "Initial")
plot(xfine, fhat(xfine, θ0 + dθ*τ), label = "Perturbed")
θf = train(θ0 + dθ*τ)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()


score, θ0, dθ = optimal_split(θf)
@show Nh(θ0), score
τ = line_search(θ0, dθ)
@show τ
figure()
plot_data()
plot(xfine, fhat(xfine, θ0), label = "Initial")
plot(xfine, fhat(xfine, θ0 + dθ*τ), label = "Perturbed")
θf = train(θ0 + dθ*τ)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()


score, θ0, dθ = optimal_split(θf)
@show Nh(θ0), score
τ = line_search(θ0, dθ)
@show τ
figure()
plot_data()
plot(xfine, fhat(xfine, θ0), label = "Initial")
plot(xfine, fhat(xfine, θ0 + dθ*τ), label = "Perturbed")
θf = train(θ0 + dθ*τ)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()


score, θ0, dθ = optimal_split(θf)
@show Nh(θ0), score
τ = line_search(θ0, dθ)
@show τ
figure()
plot_data()
plot(xfine, fhat(xfine, θ0), label = "Initial")
plot(xfine, fhat(xfine, θ0 + dθ*τ), label = "Perturbed")
θf = train(θ0 + dθ*τ)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()


score, θ0, dθ = optimal_split(θf)
@show Nh(θ0), score
τ = line_search(θ0, dθ)
@show τ
figure()
plot_data()
plot(xfine, fhat(xfine, θ0), label = "Initial")
plot(xfine, fhat(xfine, θ0 + dθ*τ), label = "Perturbed")
θf = train(θ0 + dθ*τ)
plot(xfine, fhat(xfine, θf), label = "Nh = $(Nh(θf))")
legend()

end # module
