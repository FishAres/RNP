using TSne, Clustering

function pad_z(z, args)
    zlen = size(z, 2)
    z_ = if zlen < args[:bsz]
        [z [z[:, end] for _ = 1:(args[:bsz]-zlen)]...]
    else
        z[:, 1:args[:bsz]]
    end
    z_
end

function accum_zs(model, x)
    z1s = []
    Encoder, H = model
    μ = Encoder(x)[1]
    θs, xy0, patch0 = H(μ)
    primary_nets = Encx, fx, fa, Dx, Da = gen_v2_models(θs, primaries)

    ẑ, xyt = glimpse(primary_nets, patch0, xy0)
    push!(z1s, cpu(ẑ))
    x̂ = Dx(ẑ)
    patch_t = Zygote.ignore() do
        zoom(x, xyt, "in", sampling_grid) |> flatten
    end
    out_1 = model_forward(model, ẑ) |> flatten
    out = sample_patch(out_1, xyt, sampling_grid)
    @inbounds for t = 2:args[:seqlen]
        ẑ, xyt = glimpse(primary_nets, out_1, xyt)
        push!(z1s, cpu(ẑ))
        x̂ = Dx(ẑ)
        patch_t = Zygote.ignore() do
            zoom(x, xyt, "in", sampling_grid) |> flatten
        end
        out_1 = model_forward(model, ẑ) |> flatten
        out += sample_patch(out_1, xyt, sampling_grid)
    end
    return μ, z1s
end



function get_cluster_batch(cluster, cluster_ids, all_zs)
    z = all_zs[:, cluster_ids.==cluster]
    z_ = pad_z(z, args)
    out = dropdims(model_forward(model, gpu(z_)) |> cpu, dims=3)
    return out
end

function plot_clusters(n_clusters, all_zs; savefigs=false)
    R = kmeans(all_zs, n_clusters)
    cluster_ids = assignments(R)
    p_ = [
        begin
            tmp = get_cluster_batch(i, cluster_ids, all_zs)[:, :, 1:16]
            plot_sample(tmp[end:-1:1, :, :]; xpart=4)
        end for i = 1:n_clusters
    ]
    if savefigs
        [
            savefig(p, "plots/clustering/omni_kmeans_clusters/cluster_$i.png") for
            (i, p) in enumerate(p_)
        ]
    end
    plot(p_..., size=(800, 800))
end

## ==== do tSNE


@time all_zs = let
    n_batches = 10
    zs_ = []
    for i = 1:n_batches
        μ, z1s = accum_zs(model, xs_test[i]) |> cpu
        push!(zs_, hcat(μ, z1s...))
    end
    hcat(zs_...)
end

Yt = tsne(all_zs')

## ====

function plot_tsne(Yt, n_clusters; plot_parts=true)
    R = kmeans(Yt', n_clusters)

    cluster_ids = assignments(R)

    cluster_samples = let
        sample_ims = []
        for cluster = 1:n_clusters
            z = all_zs[:, cluster_ids.==cluster]
            z_ = pad_z(z, args)
            out = dropdims(model_forward(model, gpu(z_)) |> cpu, dims=3)
            # find a representative sample
            mn = mean(out, dims=3)
            ind = argmin(mean((flatten(out) .- flatten(mn)) .^ 2, dims=1)[1, :])
            push!(sample_ims, out[:, :, ind])
        end
        sample_ims
    end

    imsize = (12, 12)
    imctr = imsize ./ 2
    p_ = plot(
        legend=false,
        axis=nothing,
        xaxis=false,
        yaxis=false,
        size=(800, 800),
    )

    for i = 1:n_clusters
        yt_ = Yt[cluster_ids.==i, :]
        scatter!(yt_[:, 1], yt_[:, 2], label=i, alpha=0.99)

        plot_parts && begin
            x, y = R.centers[:, i] .+ 7
            cluster_sample = imresize(cluster_samples[i], imsize...)[:, end:-1:1]'
            heatmap!(
                x-imctr[1]+1:x+imctr[1],
                y-imctr[2]+1:y+imctr[2],
                cluster_sample,
                color=:grays,
                clim=(0, 1),
                alpha=0.9,
            )
        end

    end
    p_
end
## ====
plot_tsne(Yt, 43; plot_parts=true)


