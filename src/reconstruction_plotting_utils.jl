using Images, Colors

function orange_on_rgb(x, a, b, c)
    a_ = a .+ x
    b_ = b .+ 0.65 .* x
    c_ = c
    a_, b_, c_
end

function orange_on_rgb_array(xs)
    xs_transpose =
        [batch([x' for x in eachslice(dropdims(z, dims=3), dims=3)]) for z in xs]
    xs_transpose = unsqueeze.(xs_transpose, 3)
    tmp_sum = orange_on_rgb(xs_transpose[end], xs_transpose[1:3]...)
    tmp_sum = cat(tmp_sum..., dims=3)
    ims = [permutedims(x, [3, 1, 2]) for x in eachslice(tmp_sum, dims=4)]
    return ims
end

function stack_ims(xs; n=8)
    n = n === nothing ? sqrt(length(xs)) : n
    rows_ = map(x -> hcat(x...), partition(xs, n))
    cat(rows_..., dims=3)
end

function impermute(x)
    a, b, c = size(x)
    if a < b && a < c
        return permutedims(x, [2, 3, 1])
    else
        return permutedims(x, [3, 1, 2])
    end
end


@inline function sample_patch_rgb(
    x,
    xy,
    sampling_grid;
    sc_offset=args[:scale_offset],
    sz=args[:img_size]
)
    ximg = reshape(x, sz..., 3, size(x)[end])
    sc_ = maprange(view(xy, 1:2, :), -1.0f0:1.0f0, sc_offset)
    xy_ = view(xy, 3:4, :)
    tr_grid = affine_grid_generator(sampling_grid, sc_, xy_)
    return grid_sample(ximg, tr_grid; padding_mode=:zeros)
end


function glimpse_seq_patches(model, x; args=args)
    patches, preds, xys, xys_1 = [], [], [], []
    function push_to_patches!(x̂, xy, out)
        push!(patches, cpu(x̂))
        push!(preds, cpu(out))
        push!(xys, cpu(xy))
    end

    r = randn(Float32, args[:π], args[:bsz]) |> gpu

    Encoder, H = model
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ)

    out1_patches, out_1, xys1 = model_forward_patches(model, ẑ)
    out = sample_patch(out_1, xyt, sampling_grid)
    push_to_patches!(out_1, xyt, out1_patches)
    push!(xys_1, xys1)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out_1, xyt)
        x̂ = Dx(ẑ)
        out1_patches, out_1, xys1 = model_forward_patches(model, ẑ)
        out += sample_patch(out_1, xyt, sampling_grid)
        push_to_patches!(out_1, xyt, out1_patches)
        push!(xys_1, xys1)
    end
    return patches, preds, xys, xys_1
end

function model_forward_patches(model, z)
    Encoder, H = model
    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    patches, xys = [], []
    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ)
    out = sample_patch(x̂, xyt, sampling_grid)
    out_ = out
    push!(patches, cpu(x̂) |> flatten)
    push!(xys, cpu(xyt))
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out, xyt)
        x̂ = Dx(ẑ)
        out = sample_patch(x̂, xyt, sampling_grid)
        push!(patches, cpu(x̂) |> flatten)
        push!(xys, cpu(xyt))
        out_ += out
    end
    return patches, out_, xys
end


function get_digit_parts(
    xs,
    batchind;
    savestring=nothing,
    format_=".png",
    resize_=(56, 56)
)
    patches, sub_patches, xys, xys1 = glimpse_seq_patches(model, xs)
    pasted_patches =
        [sample_patch(patches[i], xys[i], cpu(sampling_grid)) for i = 1:length(patches)]

    digits = orange_on_rgb_array(pasted_patches)
    digit_patches = [
        orange_on_rgb_array([
            sample_patch(sub_patch, xy, cpu(sampling_grid)) for
            (sub_patch, xy) in zip(subpatch_vec, xy_vec)
        ]) for (subpatch_vec, xy_vec) in zip(sub_patches, xys1)
    ]

    digit_im = map(clamp01nan, imresize(colorview(RGB, digits[batchind]), resize_))
    patch_ims = [
        map(clamp01nan, imresize(colorview(RGB, patch_[batchind]), resize_)) for
        patch_ in digit_patches
    ]

    if savestring !== nothing
        save(savestring * format_, digit_im)
        [
            save(savestring * "_part_$i" * format_, patch_) for
            (i, patch_) in enumerate(patch_ims)
        ]
    end
    return digits, digit_patches, digit_im, patch_ims, xys, xys1
end
