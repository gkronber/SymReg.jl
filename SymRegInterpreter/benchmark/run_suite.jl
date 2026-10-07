# Run every benchmarkable in the suite once (untuned) and report a concise
# summary.  Used by CI to validate that the benchmark suite builds and runs, and
# with a file name argument to save the medians for the benchmark history
# (github-action-benchmark); for revision comparisons use `run_benchmarks.jl`.
using BenchmarkTools

function suite_run(suite)
    results = BenchmarkGroup()
    for (k, v) in suite
        if v isa BenchmarkGroup
            results[k] = suite_run(v)
        else
            t = BenchmarkTools.run(v; samples=5)
            results[k] = t
            med = BenchmarkTools.median(t)
            @info String(k) time_ns=med.time allocs=t.allocs
        end
    end
    results
end

results = suite_run(include("benchmarks.jl"))
isempty(ARGS) || BenchmarkTools.save(ARGS[1], BenchmarkTools.median(results))
println("BENCHMARK SUITE OK")
