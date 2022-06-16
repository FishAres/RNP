using LinearAlgebra, Statistics
using IterTools: partition
using Flux, Zygote, CUDA
using Flux: batch, unsqueeze
using Plots

using ProgressMeter: Progress
using ProgressMeter

sample_z(μ, logvar, r) = @. μ + r * (exp(logvar))

kl_loss(μ, logvar) = sum(@. (exp(2.0f0 * logvar) + μ^2 - 2.0f0 * logvar - 1.0f0))


function glimpse(models, patch, xy)
    Encx, fx, fa, Dx, Da = models
    ê = Encx((flatten(patch), xy))
    x_ = fx(ê)
    xyt = Da(fa(ê))
    x_, xyt
end


function zoom_in(x, thetas, scale_offset)
    scale_dims = [1, 4]
    thetas_new = thetas .+ scale_offset
    sc_ = thetas_new[scale_dims, :]
    inv_scale = 1 ./ sc_
    thetas_new[scale_dims, :] = inv_scale
    thetas_new[5:6, :] = sc_ .* thetas_new[5:6, :]
    return sample_patch(x, thetas_new, sampling_grid)
end

function zoom(x, xy, d::String, sampling_grid)
    xout = if d == "out"
        sample_patch(x, sc_, xys, sampling_grid)
    elseif d == "in"
        zoom_in(x, xy, scale_offset)
    else
        throw(ArgumentError("zoom in or out pls"))
    end
    xout
end

# one-level model loss
function model_loss(model, x, r)
    Encoder, H = model
    μ, logvar = Encoder(x)
    z = (@.μ + r * exp(logvar)) # reparameterize
    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)
    # run init values through primary nets
    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ) # Dec(ẑ) -> patch
    # transform
    out = sample_patch(x̂, xyt, sampling_grid)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, x̂, xyt)
        x̂ = Dx(ẑ)
        out += sample_patch(x̂, xyt, sampling_grid)
    end
    return Flux.mse(flatten(out), x)
end

function plot_sample(out; xpart=8)
    out_part = partition(eachslice(cpu(out), dims=3), xpart)
    random_sample = hcat(map(x -> vcat(x...), out_part)...)
    plot_digit(random_sample)
end


# one model iteration, i.e. H(z)
function model_forward(model, z)
    Encoder, H = model
    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ)
    out = sample_patch(x̂, xyt, sampling_grid)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, x̂, xyt)
        x̂ = Dx(ẑ)
        out += sample_patch(x̂, xyt, sampling_grid)
    end
    return out
end

# 2-level model loss
function model_loss2(model, x, r)
    Encoder, H = model
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ)
    patch_t = Zygote.ignore() do
        zoom_in(x, xyt, scale_offset) |> flatten
    end

    out_1 = model_forward(model, ẑ) |> flatten
    out = sample_patch(out_1, xyt, sampling_grid)
    L1 = Flux.mse(x̂, patch_t)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out_1, xyt)
        x̂ = Dx(ẑ)
        patch_t = Zygote.ignore() do
            zoom_in(x, xyt, scale_offset) |> flatten
        end
        out_1 = model_forward(model, ẑ) |> flatten
        out += sample_patch(out_1, xyt, sampling_grid)
        L1 += Flux.mse(x̂, patch_t)
    end
    rec_loss = Flux.mse(flatten(out), flatten(x)) + (1.0f0 / args[:seqlen]) * L1
    klqp = kl_loss(μ, logvar)
    rec_loss, klqp
end

function glimpse_seq(model, x; args=args)
    # append observations to a vector for analysis
    patches, preds, xys = [], [], []
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
    patch_t = Zygote.ignore() do
        zoom_in(x, xyt, scale_offset) |> flatten
    end
    out_1 = model_forward(model, ẑ) |> flatten
    out = sample_patch(out_1, xyt, sampling_grid)
    push_to_patches!(out_1, xyt, out)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out_1, xyt)
        x̂ = Dx(ẑ)
        patch_t = Zygote.ignore() do
            zoom_in(x, xyt, scale_offset) |> flatten
        end
        out_1 = model_forward(model, ẑ) |> flatten
        out += sample_patch(out_1, xyt, sampling_grid)
        push_to_patches!(out_1, xyt, out)
    end
    return patches, preds, xys
end

function random_sample(model; z=nothing)
    z = z === nothing ? randn(Float32, args[:π], args[:bsz]) |> gpu : z
    Encoder, H = model
    θs, xy0, patch0 = H(z)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    x̂ = Dx(ẑ)
    patch_t = Zygote.ignore() do
        zoom_in(x, xyt, scale_offset) |> flatten
    end
    out_1 = model_forward(model, ẑ) |> flatten
    out = sample_patch(out_1, xyt, sampling_grid)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out_1, xyt)
        x̂ = Dx(ẑ)
        patch_t = Zygote.ignore() do
            zoom_in(x, xyt, scale_offset) |> flatten
        end
        out_1 = model_forward(model, ẑ) |> flatten
        out += sample_patch(out_1, xyt, sampling_grid)
    end
    return dropdims(out, dims=3)
end

function plot_sample(out; xpart=8)
    out_part = partition(eachslice(cpu(out), dims=3), xpart)
    random_sample = hcat(map(x -> vcat(x...), out_part)...)
    plot_digit(random_sample)
end

## ======

function update_batch(
    model,
    opt,
    ps,
    x,
    r;
    logger=nothing,
    args=args,
    clamp_grads=false
)
    loss, back = pullback(ps) do
        rec_loss, klqp = model_loss2(model, x, r)
        full_loss = args[:α] * rec_loss + args[:β] * klqp
        if logger !== nothing
            Zygote.ignore() do
                log_value(logger, "kl loss", klqp)
                log_value(logger, "rec loss", rec_loss)
                log_value(logger, "full loss", full_loss)
            end
        end
        full_loss
    end
    grad = back(1.0f0)
    clamp_grads && foreach(x -> clamp!(x, -0.2f0, 0.2f0), grad)
    for g in grad
        g[findall(isnan, g)] .= 0.0f0
    end
    Flux.update!(opt, ps, grad)
    return loss
end


function train_model(model, opt, ps, train_data; args=args, epoch=1, logger=nothing, clamp_grads=false)
    sampling_grid = get_pre_grid(args[:img_size]...) |> gpu
    rs = [randn(Float32, args[:π], args[:bsz]) for _ = 1:length(train_data)] |> gpu
    losses = zeros(length(train_data))
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    @inbounds for (i, x) in enumerate(train_data)
        loss = update_batch(
            model,
            opt,
            ps,
            x,
            rs[i];
            logger=logger,
            clamp_grads=clamp_grads
        )
        losses[i] = loss
        isnan(loss) && break
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(model, test_data)
    L = 0.0f0
    rs = [randn(Float32, args[:π], args[:bsz]) for _ = 1:length(test_data)] |> gpu
    @inbounds for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss2(model, x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end

## ===== plotting

function plot_digit(x; color=:grays, alpha=1, kwargs...)
    return heatmap(
        x[:, end:-1:1]';
        color=color,
        clim=(0, 1),
        axis=nothing,
        colorbar=false,
        alpha=alpha,
        kwargs...
    )
end

function plot_digit!(x; color=:grays, alpha=1, kwargs...)
    return heatmap!(
        x[:, end:-1:1]';
        color=color,
        clim=(0, 1),
        axis=nothing,
        colorbar=false,
        alpha=alpha,
        kwargs...
    )
end

function plot_rec(patches, preds, x, ind)
    p_patch = let
        p_ = [
            plot_digit(reshape(patch, args[:img_size]..., size(x)[end])[:, :, ind])
            for patch in patches
        ]
        plot(p_...)
    end
    p_rec = plot_digit(reshape(preds[end], args[:img_size]..., size(x)[end])[:, :, ind])
    px = plot_digit(reshape(cpu(x), args[:img_size]..., size(x)[end])[:, :, ind])
    plot(px, p_patch, p_rec, layout=(1, 3))
end

function plot_recs(model, x, inds)
    patches, preds, xys = glimpse_seq(model, x)
    p_ = [plot_rec(patches, preds, x, ind) for ind in inds]
    plot(p_..., layout=(length(inds), 1), size=(400, 800))
end

## ==== Encoder architecture

"'top-level' encoder"
function get_encoder(; vae=false, z_size=args[:π], conv_channels=32, enc_lsize=128)
    enc1 = Chain(
        x -> reshape(x, args[:img_size]..., 1, size(x)[end]),
        Conv((5, 5), 1 => conv_channels, relu),
        MaxPool((2, 2)),
        BasicBlock(conv_channels => conv_channels, +),
        BasicBlock(conv_channels => conv_channels, +),
        BasicBlock(conv_channels => conv_channels, +),
        BasicBlock(conv_channels => conv_channels, +),
        Conv((5, 5), conv_channels => 16, relu),
        flatten,
    )
    # * get output size when convolutions applied to (args[:mn_size]) image
    # * arbitrary batch size 32 (shouldn't affect anything)
    dense_lsz = Flux.outputsize(enc1, (args[:img_size]..., 1, 32))[1]

    enc_head =
        vae ? Split(
            Dense(enc_lsize, z_size), # μ
            Dense(enc_lsize, z_size), # log var.
        ) : Dense(enc_lsize, z_size, elu)

    full_encoder = Chain(
        enc1,
        Dense(dense_lsz, enc_lsize),
        BatchNorm(enc_lsize, relu),
        Dense(enc_lsize, enc_lsize),
        BatchNorm(enc_lsize, elu),
        Dense(enc_lsize, enc_lsize),
        BatchNorm(enc_lsize, relu),
        Dense(enc_lsize, enc_lsize),
        BatchNorm(enc_lsize, elu),
        enc_head,
    )
    full_encoder
end
