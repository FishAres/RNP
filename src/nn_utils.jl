using Flux, Zygote
using TensorBoardLogger, Logging
using JLD2
using BSON

## =====
"For multiple output heads"
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

## ======

dense_plus(in_sz, lsz; f=elu, l2=BatchNorm) = Chain(Dense(in_sz, lsz), l2(lsz, f))

struct ResidualBlock
    block::Any
end
Flux.@functor ResidualBlock
(b::ResidualBlock)(x) = relu.(b.block(x))

function BasicBlock(channels::Pair{Int64,Int64}, connection; stride::Int64=1)
    layer = Chain(
        Conv((3, 3), channels; stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv((3, 3), channels[2] => channels[2]; pad=1, bias=false),
        BatchNorm(channels[2]),
    )
    return Chain(SkipConnection(layer, connection), x -> relu.(x))
end

## ===== saving etc

"horrible function that needs recursion - handles depth up to 1"
Zygote.@nograd function get_param_sizes(model)
    ps = []
    for m in model.layers
        if hasproperty(m, :weight)
            push!(ps, size(m.weight))
            if hasproperty(m, :bias) && typeof(m.bias) !== Flux.Zeros
                # try
                push!(ps, size(m.bias))
                # catch y
                # if isa(y, MethodError)
                # println("no bias to see here")
                # nothing
                # end
                # end
            end
        else
            for m_ in m.layers
                if hasproperty(m_, :weight)
                    push!(ps, size(m_.weight))
                    # if hasproperty(m_, :bias)
                    if hasproperty(m_, :bias) && typeof(m_.bias) !== Flux.Zeros
                        # try
                        push!(ps, size(m_.bias))
                        # catch y
                        # if isa(y, MethodError)
                        # println("no bias to see here")
                        # nothing
                        # end
                        # end
                    end
                end
            end
        end
    end
    return ps
end

function save_model(model, savestring; local_=true)
    model = cpu(model)
    if local_
        full_str = "saved_models/" * savestring * ".bson"
    else
        full_str = savestring * ".bson"
    end
    BSON.@save full_str model
    return println("saved at $(full_str)")
end
