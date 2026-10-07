# SymReg.jl

Julia packages for fitting and comparing symbolic regression models, shared by
symbolic regression algorithms such as [NeoGP](https://github.com/gkronber/NeoGP.jl):

| Package | Contents |
|---|---|
| [`SymRegInterpreter`](SymRegInterpreter/) | Models given as expressions `(x, p) -> …` or as linear programs of instructions; allocation-free evaluation with reverse-mode gradients and Jacobians. |
| [`SymRegLikelihoods`](SymRegLikelihoods/) | Likelihoods (Gaussian and Laplace with known, profiled or intrinsic-scatter noise, errors in the inputs, marginalized normal regression) and parameter fitting with Levenberg-Marquardt ([GeodesicLM](https://github.com/gkronber/GeodesicLM.jl)) or NLopt; information matrix, confidence and prediction intervals. |
| [`SymRegModelSelection`](SymRegModelSelection/) | Model-selection criteria: description length, fractional Bayes factor, AIC, BIC; rational constant snapping. |

## Installation

The packages are not registered.  With Julia 1.13 or newer, add them from this
repository in one call (they refer to each other within the repository):

```julia
using Pkg
url = "https://github.com/gkronber/SymReg.jl"
Pkg.add([PackageSpec(; url, subdir = "SymRegInterpreter"),
         PackageSpec(; url, subdir = "SymRegLikelihoods"),
         PackageSpec(; url, subdir = "SymRegModelSelection")])
```

## Example

```julia
using SymRegInterpreter, SymRegLikelihoods, SymRegModelSelection
using Random

rng = Xoshiro(1)
X = reshape(collect(range(0.1, 5.0; length = 50)), :, 1)   # one observation per row
y = 2.0 .* exp.(-0.7 .* X[:, 1]) .+ 0.05 .* randn(rng, 50)

model = Model(Float64, :((x, p) -> p[1] * exp(p[2] * x[1])), 2)
l = GaussianProfiledLikelihood(X, y)          # Gaussian noise with unknown variance
r = optimize(l, model; rng)                   # r.x: fitted [p[1], p[2], σ²]

dl, nll, funcComp, paramComp = description_length(l, model, r.x)
BIC(l, model, r.x)
```

Every exported and public function has a docstring, e.g. `?optimize` or
`?description_length`.

## Development

Run the tests of a package with

```bash
julia --project=SymRegLikelihoods -e 'using Pkg; Pkg.test()'
```

Each package has a benchmark suite in its `benchmark/` folder, and
`coverage/run_coverage.jl` collects the test coverage of all three packages.

## License

MIT, see [LICENSE](LICENSE).
