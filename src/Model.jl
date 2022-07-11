module Model

export λ, Nh, fhat, loss, train, optimal_split, line_search, target_split

using LinearAlgebra
using Main.Data
using Flux
using Optim
using ForwardDiff: gradient, hessian

# activation
σ(x) = 1/(1 + exp(-x))
# Regularization parameter
const λ = 1e-4

Nh(θ) = div(length(θ) - 1, 3)
W1(θ, Nh) = reshape(θ[1:Nh], (Nh, 1))
B1(θ, Nh) = θ[Nh+1:2*Nh]
W2(θ, Nh) = reshape(θ[2*Nh+1:3*Nh], (1,Nh))
B2(θ, Nh) = θ[end]

function logit(x, θ)
    n = Nh(θ)
    return W2(θ, n) * σ.( W1(θ,n) * x .+ B1(θ, n)) .+ B2(θ, n)
end

fhat(x, θ) = vec(σ.( logit(x', θ)))

regularization(θ, λ) = λ > 0 ? λ * sum( abs2.(θ)) : 0
data_loss(θ) = Flux.Losses.logitbinarycrossentropy(logit(xdata, θ), ydata)
loss(θ, λ = λ) = data_loss(θ) + regularization(θ, λ)

function train(θ, λ = λ, verbose = true)
    result = optimize(θ->loss(θ, λ), θ, LBFGS())
    if verbose
        @show result
    end
    return result.minimizer
end

# Parameters after splitting on Neuron i
function split(θ, i)
    n = Nh(θ)
    w1 = θ[1:n]
    b1 = θ[n+1:2*n]
    w2 = θ[2*n+1:3*n]
    b2 = θ[end]
    w2[i] = w2[i]/2
    return [w1; w1[i]; b1; b1[i]; w2; w2[i]; b2]
end

# Returns indices in θ associated with neuron i
indices(i, Nh) = [i, # W1
                  Nh + i, # B1
                  2*Nh + i # W2
                  ]

function optimal_split(θ0)
    nh = Nh(θ0)
    θ_next = zeros( 3*(nh+1) + 1)
    dθ = zeros(3*(nh+1) + 1)
    max_score = 0
    for i = 1:Nh(θ0)
        θ = split(θ0, i)
        hessian_indices = [indices(i, nh+1); indices(nh+1, nh+1)]
        h = hessian(θ -> loss(θ,0), θ)[hessian_indices, hessian_indices]
        e = eigen(h)
        score = -minimum(e.values)
        @info "Trying Neuron $i split, score = $(score)"
        if score > max_score
            @info "Neuron $i is new optimal"
            max_score = score
            θ_next .= θ
            dθ .= 0
            dθ[hessian_indices] .= e.vectors[:, argmin(e.values)] ./ sqrt( abs(score))
        end
    end
    return max_score, θ_next, dθ
end
            
function target_split(θ0, i)
    nh = Nh(θ0)
    θ = split(θ0, i)
    dθ = zero(θ)
    hessian_indices = [indices(i, nh+1); indices(nh+1, nh+1)]
    h = hessian(θ -> loss(θ,0), θ)[hessian_indices, hessian_indices]
    e = eigen(h)
    score = -minimum(e.values)
    dθ[hessian_indices] .= e.vectors[:, argmin(e.values)] ./ sqrt( abs(score))
    return -minimum(e.values), θ, dθ
end

function line_search(θ, dθ, τs = range(0,2,1001))
    losses = [loss(θ + dθ*τ) for τ in τs]
    @info "Minimum loss in linear search: $(minimum(losses))"
    return τs[argmin(losses)]
end

end # module
