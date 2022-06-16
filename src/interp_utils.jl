using LinearAlgebra, Statistics
using Flux, Zygote, NNlib
using Flux: unsqueeze

function get_pre_grid(width, height; args=args)
    x = collect(LinRange(-1, 1, width))
    y = collect(LinRange(-1, 1, height))
    x_t_flat = reshape(repeat(x, height), 1, height * width)
    y_t_flat = reshape(repeat(transpose(y), width), 1, height * width)
    sampling_grid = vcat(x_t_flat, y_t_flat)
    sampling_grid = reshape(
        transpose(repeat(transpose(sampling_grid), args[:bsz])),
        2,
        size(x_t_flat)[2],
        args[:bsz],
    )
    return collect(Float32.(sampling_grid))
end

# caution - scale_offset is provided from outside the function input
function affine_grid_generator(sampling_grid, thetas; args=args, sz=args[:img_size])
    bsz = size(thetas)[end]
    thetas = isa(thetas, CuArray) ? thetas .+ scale_offset : thetas .+ cpu(scale_offset)
    thetas = reshape(thetas, 2, 3, bsz)
    tr_grid = batched_mul(thetas, sampling_grid)
    return reshape(tr_grid, 2, sz..., bsz)
end

function sample_patch(x, xy, sampling_grid; sz=args[:img_size])
    ximg = reshape(x, sz..., 1, size(x)[end])
    tr_grid = affine_grid_generator(sampling_grid, xy; sz=sz)
    grid_sample(ximg, tr_grid; padding_mode=:zeros)
end