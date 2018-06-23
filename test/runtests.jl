using FluxExtensions
using Base.Test
using Flux
using ForwardDiff
using CoordinateTransformations

@testset "Flux extensions" begin
    srand(1)
    models = [
        Chain(
            AffineMap(randn(1, 1), randn(1)),
            Dense(1, 1, elu),
            Dense(1, 1, elu),
        ),
        Chain(
            Dense(1, 10, elu),
            Dense(10, 1, elu),
            AffineMap(randn(1, 1), randn(1)),
        ),
        Chain(
            Dense(1, 10, elu),
            AffineMap(randn(10, 10), randn(10)),
            Dense(10, 1, elu),
        ),
    ]
    for m in models
        mp = FluxExtensions.TangentPropagator(m)
        p = FluxExtensions.untrack(m)

        for i in 1:100
            x = randn(1)
            y = m(x)
            y2, J = mp(x)
            @test FluxExtensions.value(y) ≈ FluxExtensions.value(y2)
            @test p(x) ≈ FluxExtensions.value(y)
            @test FluxExtensions.value(J) ≈ ForwardDiff.jacobian(p, x)
        end

        lf = (x, y, J) -> begin
            ŷ, Ĵ = mp(x)
            Flux.mse(ŷ, y) + Flux.mse(Ĵ, J)
        end
        train_data = [
            ([1.0], [1.2], [1.5])
        ]
        opt = Flux.Optimise.Momentum(params(mp))
        for i in 1:1000
            Flux.train!(lf, train_data, opt)
        end
        ŷ, Ĵ = mp([1.0])
        @test FluxExtensions.value(ŷ) ≈ [1.2]
        @test FluxExtensions.value(Ĵ) ≈ [1.5]

    end
end
