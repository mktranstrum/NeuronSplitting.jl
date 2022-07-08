module Data

export xfine, xdata, ydata, plot_data, f

using PyPlot
import Random: seed!
seed!(203894)

σ(x) = 1/(1 + exp(-x))

sample(items, weights) = items[findfirst(cumsum(weights) .> rand())]

# Probability of class 1
f(x) = σ(-25 * (x + 1.5)) + σ(25 * (x + 0.75)) + σ(-25 * (x -  0)) + σ(25 * (x - 0.75)) + σ(-25 * (x - 1.5)) - 2
bernoulli(p) = [p, 1-p]
sample(x) = sample([1, 0], bernoulli(f(x)))

const xfine = Array(range(-2,2,1001))
const xdata = reshape(rand(100)*4 .- 2.0, (1,100))
const ydata = sample.(xdata)
const fdata = f.(xdata)
const ffine = f.(xfine') |> vec

function plot_data()    
    plot(xfine, ffine, "k-")
    plot(vec(xdata), vec(ydata), "ro")
    nothing
end


end # module
