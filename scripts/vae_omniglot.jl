using DrWatson
@quickactivate "RNP_clean"
ENV["GKSwstype"] = "nul" # to make plots in headless mode

using Distributions, Random
using MLDatasets
using Plots

include(srcdir("hypernet_utils.jl"))
include(srcdir("interp_utils.jl"))
include(srcdir("model_utils.jl"))
include(srcdir("nn_utils.jl"))

## =======

args = Dict(
    :img_size => (28, 28),
    :bsz => 64, # batch size
    :π => 96, # latent dim (z) size
    :seqlen => 4, # number of glimpses
    :lsz => 64, # layer size of certain NNs
    :esz => 32, # rnn input size
    :α => 1.0f5, # reconstruction loss term
    :β => 0.5f0, # KL loss term
    :λ => 0.002f0, # weight regularization
    :lr => 4e-5,
)

## =====
all_chars = load("data/exp_pro/omniglot.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
x_batches = batch.(partition(xs, args[:bsz]))


ntrain, ntest = 286, 15
xs_train = flatten.(x_batches[1:ntrain] |> gpu)
xs_test = flatten.(x_batches[ntrain+1:ntrain+ntest] |> gpu)
x = xs_test[1]

## =====
xy_grid = get_pre_grid(args[:img_size]...)
grid_ones = ones(Float32, 1, 784, args[:bsz])
const sampling_grid = vcat(xy_grid, grid_ones) |> gpu

## =====
# scale offset for shrinking an image
# e.g. offset = 3 => image scale in 1/range(3-1,3+1) = (2,4)
const scale_offset = let
    tmp = zeros(Float32, 6, 1) |> gpu
    tmp[[1, 4], :] .= 3.3f0
    tmp
end

## =====

th_size = 6 # number of parameters for spatial transformer (6)

# patch encoder (x̂(t), a(t) -> RNN input)
enc_primary = Chain(
    Parallel(vcat, Dense(784, 64), Dense(th_size, 64)),
    Dense(128, 64),
    Dense(64, 64),
    Dense(64, args[:esz]),
)
# z -> x̂
Dx_p = dec_x(args[:π], args[:lsz], 784)
# h_policy -> a(t+1)
Da_p = dec_x(args[:π], 32, th_size)

primaries = enc_primary, Dx_p, Da_p
# parameter sizes for each primary network
θ_sizes = get_θ_sizes(primaries, args[:π], args[:esz])

H =
    Chain(
        LayerNorm(args[:π]),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Split(
            Split([Dense(64, θk, bias=false) for θk in θ_sizes]...),
            # fa initialization
            Chain(Dense(64, 64), LayerNorm(64, elu), Dense(64, th_size, sin, bias=false)),
            # fx initialization
            Chain(Dense(64, 64), LayerNorm(64, elu), Dense(64, 784, relu, bias=false)),
        ),
    ) |> gpu

Encoder = get_encoder(; vae=true) |> gpu

model = Encoder, H

ps = Flux.params(model)

## =====
opt = ADAM(args[:lr])

## =====
@inbounds for epoch = 1:20
    ls = train_model(model, opt, ps, xs_train; epoch=epoch, clamp_grads=true)
    let
        inds = sample(1:args[:bsz], 6; replace=false)
        p = plot_recs(model, rand(xs_test), inds)
        display(p)

        if epoch % 5 == 0
            out_ = random_sample(model)
            prs = plot_sample(out_)
            display(prs)
        end
    end

    L = test_model(model, xs_test)

    if epoch % 50 == 0
        model_path =
            joinpath(save_dir, savename_(args)) * "_" * model_alias * "_epoch$(epoch)"
        save_model(cpu(model), model_path; local_=false)
    end
end

## =====
