# Run every benchmarkable in the suite once (untuned) and report a concise
# summary.  Used by CI to validate that the benchmark suite builds and runs;
# for detailed revision comparisons use `run_benchmarks.jl` (AirspeedVelocity).
using BenchmarkTools

function suite_run!(suite; per_bench_sec::Float64 = 0.1)
    for (k, v) in suite
        if v isa BenchmarkGroup
            suite_run!(v; per_bench_sec)
        else
            t = BenchmarkTools.run(v; samples=5)
            med = BenchmarkTools.median(t)
            @info String(k) time_ns=med.time allocs=t.allocs
        end
    end
    nothing
end

suite_run!(include("benchmarks.jl"))
println("BENCHMARK SUITE OK")
