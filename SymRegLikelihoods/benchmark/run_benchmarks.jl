# Run the SymRegLikelihoods benchmark locally using AirspeedVelocity.
# Compares two git revisions of the SymReg.jl monorepo and prints a markdown
# table with median runtime and allocation differences.
#
# AirspeedVelocity expects the package to be a git repository, so each revision
# is copied into a standalone repo tagged with the revision string.
#
# Usage (from the package dir):
#   julia --project=benchmark benchmark/run_benchmarks.jl
#   julia --project=benchmark benchmark/run_benchmarks.jl HEAD dirty
#   julia --project=benchmark benchmark/run_benchmarks.jl <rev1> <rev2>

using AirspeedVelocity
using Pkg
using TOML

const PACKAGE_NAME = "SymRegLikelihoods"
const PACKAGE_DIR  = realpath(joinpath(@__DIR__, ".."))
const REPO_ROOT    = realpath(joinpath(@__DIR__, "..", ".."))   # SymReg.jl git root
const GEODESICLM_DIR = normpath(joinpath(REPO_ROOT, "..", "GeodesicLM.jl"))
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
    # Benchmark against the SymRegInterpreter of the same revision.
    sibling = joinpath(dir, "SymRegInterpreter")
    if !isdir(sibling)
        if rev == "dirty"
            cp(joinpath(REPO_ROOT, "SymRegInterpreter"), sibling; force=true)
        else
            run(pipeline(`git -C $REPO_ROOT archive $rev SymRegInterpreter`, `tar -x -C $dir`))
        end
    end
    # Point `[sources]` at the sibling and at the local GeodesicLM checkout (if any).
    # Without one, a revision keeps its GeodesicLM URL; older revisions referenced
    # a checkout by a relative path and get the first release instead.
    toml = joinpath(pkgdir, "Project.toml")
    project = TOML.parsefile(toml)
    sources = get!(project, "sources", Dict{String,Any}())
    sources["SymRegInterpreter"] = Dict("path" => sibling)
    if isdir(GEODESICLM_DIR)
        sources["GeodesicLM"] = Dict("path" => GEODESICLM_DIR)
    elseif haskey(get(sources, "GeodesicLM", Dict()), "path")
        sources["GeodesicLM"] = Dict("url" => "https://github.com/gkronber/GeodesicLM.jl", "rev" => "v0.1.0")
    end
    open(io -> TOML.print(io, project), toml, "w")
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
