using DrWatson
@quickactivate "RNP"
ENV["GKSwstype"] = "nul" # to make plots in headless mode

using Distributions, Random
using Flux, Zygote, CUDA
using MLDatasets
using Plots

include(srcdir("hypernet_utils.jl"))
include(srcdir("interp_utils.jl"))
include(srcdir("model_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("reconstruction_plotting_utils.jl"))

## =======

args = Dict(
    :img_size => (28, 28),
    :bsz => 64, # batch size
    :π => 32, # latent dim (z) size - 32 for (Fashion)MNIST, 96 for omniglot
    :seqlen => 4, # number of glimpses
    :lsz => 64, # layer size of certain NNs
    :esz => 32, # rnn input size
    :α => 1.0f5, # reconstruction loss term
    :β => 0.5f0, # KL loss term
    :λ => 0.002f0, # weight regularization
    :lr => 4e-5,
)

## =====
train_digits, test_digits, test_labels = let
    train_digits, _ = MNIST.traindata(Float32)
    test_digits, test_labels = MNIST.testdata(Float32)
    xs_train =
        batch.(partition(shuffle(collect(eachslice(train_digits, dims=3))), args[:bsz]))
    test_shuffle_inds =
        sample(1:length(test_labels), length(test_labels), replace=false)
    xs_test =
        batch.(
            partition(
                collect(eachslice(test_digits, dims=3))[test_shuffle_inds],
                args[:bsz],
            ),
        )
    ys_test = batch.(partition(test_labels[test_shuffle_inds], args[:bsz]))
    xs_train, xs_test, ys_test
end

## +=======
xs_train = flatten.(train_digits) |> gpu
xs_test = flatten.(test_digits) |> gpu
ys_test = test_labels
x = xs_test[1]

## =====
xy_grid = get_pre_grid(args[:img_size]...)
grid_ones = ones(Float32, 1, 784, args[:bsz])
const sampling_grid = vcat(xy_grid, grid_ones) |> gpu

## =====
# hack - scale offset for shrinking an image
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

modelpath = "saved_models/bsz=64_esz=32_lr1=4e-5_lsz=64_patch=false_seqlen=4_α=80000.0_β=0.5_λ=0.002_π=32_vae_mnist_rotation_epoch400.bson"

model = load(modelpath)[:model] |> gpu
Encoder, H = model

## ====

# plot pretty things!
ind = 0
begin
    ind += 1
    x = xs_test[1]
    digits, digit_patches, digit_im, patch_ims, xys, xys1 = get_digit_parts(x, ind)
    digit_im
end

