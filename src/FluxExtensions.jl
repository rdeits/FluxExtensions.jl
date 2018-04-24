__precompile__()

module FluxExtensions

using Flux
using ForwardDiff
using CoordinateTransformations

const value = Flux.Tracker.data

activation(σ) = x -> σ.(x)

plain(layer::Dense) = activation(layer.σ) ∘ AffineMap(value(layer.W), value(layer.b))
plain(t::Transformation) = t
plain(f::Function) = f
plain(chain::Chain) = reduce(∘, identity, plain.(reverse(chain.layers)))

struct TangentPropagator{F <: Function, C}
    f::F
    layer::C
end

function TangentPropagator(chain::Chain)
    f = reduce(∘, identity, _propagate_tangent.(reverse(chain.layers)))
    TangentPropagator(x -> f((x, eye(length(x)))), chain)
end

(p::TangentPropagator)(x) = p.f(x)

Flux.params(p::TangentPropagator) = Flux.params(p.layer)

function _propagate_tangent(f)
    (xJ) -> begin
        (f(xJ[1]), ForwardDiff.jacobian(f, value(xJ[1])) * xJ[2])
    end
end

function _propagate_tangent(f::Dense)
    xJ -> begin
        x, J = xJ
        y = f.W * x + f.b
        gσ = ForwardDiff.derivative.(f.σ, y)
        (f(x), gσ .* f.W * J)
    end
end

struct Attention{T1, T2}
    signals::T1
    weights::T2
end

function avoid_zero(x, tol=1e-6)
    if abs(x) < tol
        if x < 0
            return x + (-tol - x)
        else
            return x + (tol - x)
        end
    end
    return x
end

function (a::Attention)(x)
    sum(a.weights(x) .* a.signals(x), 1)
end

Flux.params(a::Attention) = vcat(params(a.signals), params(a.weights))

function TangentPropagator(a::Attention)
    t1 = TangentPropagator(a.signals)
    t2 = TangentPropagator(a.weights)
    function f(x)
        x1, J1 = t1(x)
        x2, J2 = t2(x)
        y = a(x)
        J = sum(x1 .* J2, 1) .+ sum(x2 .* J1, 1)
        y, J
    end
    TangentPropagator(f, a)
end

TangentPropagator(d::Dense) = TangentPropagator(Chain(d))

end
