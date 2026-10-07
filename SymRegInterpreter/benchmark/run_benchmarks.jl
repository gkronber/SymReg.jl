# Run the SymRegInterpreter benchmark locally using AirspeedVelocity.
# Compares two git revisions of the SymReg.jl monorepo and prints a markdown
# table with median runtime and allocation differences.
#
# The shared packages live in subdirectories of the SymReg.jl git repository, not
# as git repos themselves, and AirspeedVelocity expects each benchmarked package
# to BE a git repository (it names results by `rev` and `Pkg.add`'s a path+rev
# spec).  So each revision is materialised as a *self-contained* git repo of the
# package, tagged with the revision string, and benchmarked from there.
#
# Usage (from the package dir):
#   julia --project=benchmark benchmark/run_benchmarks.jl
#   julia --project=benchmark benchmark/run_benchmarks.jl HEAD dirty
#   julia --project=benchmark benchmark/run_benchmarks.jl <rev1> <rev2>

using AirspeedVelocity
using Pkg

const PACKAGE_NAME = "SymRegInterpreter"
const PACKAGE_DIR  = realpath(joinpath(@__DIR__, ".."))
const REPO_ROOT    = realpath(joinpath(@__DIR__, "..", ".."))   # SymReg.jl git root
const OUTPUT_DIR   = mkpath(joinpath(@__DIR__, "results"))
const BENCH_SCRIPT = joinpath(@__DIR__, "benchmarks.jl")

# ---------- helpers ----------

# Materialise a revision of a package as a standalone git repo tagged `rev`.
function snapshot_repo(dir, package, rev)
    pkgdir = joinpath(dir, package)
    if rev == "dirty"
        cp(joinpath(REPO_ROOT, package), pkgdir; force=true)   # incl. uncommitted changes
    else
        run(pipeline(`git -C $REPO_ROOT archive $rev $package`, `tar -x -C $dir`))
    end
    # SymRegLikelihoods depends on SymRegInterpreter via a relative `[sources]`
    # path; give it a sibling so that signature resolves inside the snapshot.
    if package == "SymRegLikelihoods" || package == "SymRegInterpreter"
        sibling = joinpath(dir, "SymRegInterpreter")
        if !isdir(sibling)
            if rev == "dirty"
                cp(joinpath(REPO_ROOT, "SymRegInterpreter"), sibling; force=true)
            else
                run(pipeline(`git -C $REPO_ROOT archive $rev SymRegInterpreter`, `tar -x -C $dir`))
            end
        end
    end
    # standalone git repo, tagged so `Pkg.add(path=..., rev=rev)` resolves
    run(`git -C $pkgdir init -q`)
    run(`git -C $pkgdir add -A`)
    run(`git -C $pkgdir -c user.name=bench -c user.email=bench@localhost commit -q -m "snapshot $rev"`)
    run(`git -C $pkgdir tag $rev`)
    return pkgdir
end

# ---------- main ----------

# Parse revision arguments (default: compare HEAD vs dirty)
revs = if length(ARGS) >= 2
    ARGS[1:2]
else
    ["HEAD", "dirty"]
end

println("# Benchmark comparison: $(revs[1]) vs $(revs[2])\n")

paths = Dict{String,String}()
snapshotted = String[]   # temp root dirs to clean up

try
    for rev in revs
        snapshot_root = mktempdir(; cleanup=true)
        paths[rev] = snapshot_repo(snapshot_root, PACKAGE_NAME, rev)
        push!(snapshotted, snapshot_root)
    end

    specs = [
        PackageSpec(; name=PACKAGE_NAME, path=paths[rev], rev=rev)
        for rev in revs
    ]

    @info "Running benchmarks for: $(revs[1]) vs $(revs[2])"
    AirspeedVelocity.benchmark(
        specs;
        script=BENCH_SCRIPT,
        output_dir=OUTPUT_DIR,
        tune=false,
    )

    combined_results = AirspeedVelocity.load_results(specs; input_dir=OUTPUT_DIR)

    println("\n## Median Runtime\n")
    println(AirspeedVelocity.create_table(
        combined_results;
        key="median",
        add_ratio_col=true,
    ))

    println("\n## Memory / Allocations\n")
    println(AirspeedVelocity.create_table(
        combined_results;
        key="memory",
        add_ratio_col=true,
    ))

catch e
    @error "Benchmark run failed: $e"
    rethrow(e)

finally
    for dir in snapshotted
        rm(dir; force=true, recursive=true)
    end
end
