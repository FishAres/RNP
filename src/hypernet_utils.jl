
"to avoid having to reshape everything manually"
@inline bmul(a, b) = dropdims(batched_mul(unsqueeze(a, 1), b); dims=1)

"RNN parameterized by Wh, Wx, b, with initial state h"
function RN(Wh, Wx, b, h, x; f=elu)
    h = f.(bmul(h, Wh) + bmul(x, Wx) + b)
    return h, h
end


"(θ: b, Wh, Wx, h) -> Recur(RNN(..., h))"
function ps_to_RN(θ; rn_fun=RN, args=args, f_out=elu)
    Wh, Wx, b, h = θ
    ft = (h, x) -> rn_fun(Wh, Wx, b, h, x; f=f_out)
    return Flux.Recur(ft, h) # stateful module with inital state h
end

"size for RNN Wh, Wx, b"
get_rnn_θ_sizes(esz, hsz) = esz * hsz + hsz^2 + 2 * hsz

function rec_rnn_θ_sizes(esz, hsz)
    [hsz^2, esz * hsz, hsz, hsz]
end

function get_rn_θs(rnn_θs, esz, hsz)
    fx_sizes = rec_rnn_θ_sizes(esz, hsz)
    @assert sum(fx_sizes) == size(rnn_θs, 1)
    fx_inds = [0; cumsum(fx_sizes)]
    # split rnn_θs vector according to cumulative fx_inds
    Wh_, Wx_, b, h =
        collect(rnn_θs[fx_inds[ind-1]+1:fx_inds[ind], :] for ind = 2:length(fx_inds))
    Wh = reshape(Wh_, hsz, hsz, size(Wh_)[end])
    Wx = reshape(Wx_, args[:esz], args[:π], size(Wx_)[end])
    Wh, Wx, b, h
end

## ==== hyper-Dense layer

struct HyDense
    W::Any
    b::Any
    HyDense(W, b) = new(W, b)
end

HyDense(W) = HyDense(W, Flux.Zeros())

(m::HyDense)(x) = bmul(x, m.W) + m.b

## =====

dec_x(hsz, lsz, out_sz) = Chain(
    Dense(hsz, lsz),
    Dense(lsz, lsz),
    Dense(lsz, lsz),
    Dense(lsz, out_sz, bias=false),
)

"Get the size of all the parameters of model m"
get_param_sizes(m) = size.(Flux.params(m))

"hacky function to mirror the patch encoder"
function get_enc(enc_primary, θs_enc)
    p_sizes = get_param_sizes(enc_primary)
    ws_enc = [get_θw(θs_enc, i, p_sizes) for i = 1:length(p_sizes)]
    Encx = Chain(
        Parallel(
            vcat,
            get_dense_bn(ws_enc, 1; act_norm=LayerNorm, f=elu),
            get_dense_bn(ws_enc, 3; act_norm=LayerNorm, f=elu),
        ),
        get_dense_bn(ws_enc, 5; act_norm=LayerNorm, f=elu),
        get_dense_bn(ws_enc, 7; act_norm=LayerNorm, f=elu),
        get_dense_bn(ws_enc, 9; act_norm=LayerNorm, f=elu),
    )
    return Encx
end

"get the part of params θ that corresponds to module i"
function get_θw(θ, ind, p_sizes)
    size_prods = prod.(p_sizes)
    cum_sizes = cumsum(size_prods)
    tot_sizes = [0; cum_sizes...]
    θi = θ[(tot_sizes[ind]+1):tot_sizes[ind+1], :]
    return reshape(θi, reverse(p_sizes[ind])..., size(θi)[end])
end

"return a dense network with a normalization layer from ''weights''"
function get_dense_bn(weights, k; act_norm=BatchNorm, f=elu)
    lsize = size(weights[k+1], 1)
    m_ = HyDense(weights[k], weights[k+1])
    Chain(m_, act_norm(lsize, f))
    return m_
end

function get_psize_prods(model)
    p_sizes = get_param_sizes(model)
    return prod.(p_sizes)
end

"return a Chain of Dense layers parameterized by (#params x batch size) matrix θ"
function get_H_model(p_sizes, θ; act_norm=BatchNorm, f=elu, f_out=relu)
    weights = [get_θw(θ, i, p_sizes) for i = 1:length(p_sizes)]
    m = Chain(
        [get_dense_bn(weights, k; act_norm=act_norm, f=f) for k in (1, 3, 5)]...,
        HyDense(weights[end]),
        x -> f_out.(x),
    )
    return m
end

"generates a primary network model from parameters θ"
function get_H_model(model::Flux.Chain, θ; act_norm=BatchNorm, f=elu, f_out=elu)
    p_sizes = get_param_sizes(model)
    get_H_model(p_sizes, θ; act_norm=act_norm, f=f, f_out=f_out)
end



"calculate parameter sizes for primary networks"
function get_θ_sizes(primaries, hsz, esz)
    enc_primary, Dx_p, Da_p = primaries
    θenc = sum(prod.(get_param_sizes(enc_primary)))
    θfx = get_rnn_θ_sizes(esz, hsz)
    θfa = get_rnn_θ_sizes(esz, hsz)
    θdec_x = sum(prod.(get_param_sizes(Dx_p)))
    θdec_a = sum(prod.(get_param_sizes(Da_p)))
    return (θenc, θfx, θfa, θdec_x, θdec_a)
end


"""
Generate Encoder, fx, fa, patch decoder, pos decoder
from θs (hypernet output)
"""
function gen_v2_models(θs, primaries)
    enc_primary, Dx_p, Da_p = primaries
    Encx = get_enc(enc_primary, θs[1])
    θs_fx = get_rn_θs(θs[2], args[:esz], args[:π])
    fx = ps_to_RN(θs_fx)
    θs_fa = get_rn_θs(θs[3], args[:esz], args[:π])
    fa = ps_to_RN(θs_fa)
    Dx = get_H_model(Dx_p, θs[4]; act_norm=LayerNorm, f_out=relu)
    Da = get_H_model(Da_p, θs[5]; act_norm=LayerNorm, f_out=sin)
    Encx, fx, fa, Dx, Da
end
