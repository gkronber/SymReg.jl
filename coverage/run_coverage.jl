# Collect line coverage of the unit tests of the SymReg.jl packages and write
# LCOV tracefiles that can be analysed with the `lcov` tools (or uploaded to
# Codecov / Coveralls in CI).
#
# Each package is tested in a separate Julia process with `Pkg.test`, tracking
# only the package's own `src/` directory, so the coverage of a package reflects
# its own test suite.  The resulting `.cov` files are converted
# with CoverageTools.jl, which also parses the sources and marks lines of
# functions that were never compiled as not covered (Julia's native LCOV output
# omits them).  Stale and generated `.cov` files are removed from `src/`.
#
# Outputs (paths in the tracefiles are relative to the repository root):
#   coverage/results/<Package>.info   one tracefile per package
#   coverage/results/lcov.info        merged tracefile of all processed packages
#   coverage/results/html/            HTML report (only with `--html`, needs `genhtml`)
#
# Usage (from the repository root):
#   julia --project=coverage -e 'using Pkg; Pkg.instantiate()'   # once
#   julia --project=coverage coverage/run_coverage.jl
#   julia --project=coverage coverage/run_coverage.jl SymRegInterpreter --html
#
# Analysis with lcov:
#   lcov --summary coverage/results/lcov.info
#   lcov --list coverage/results/lcov.info
#   genhtml coverage/results/lcov.info --output-directory coverage/results/html
#
# The script exits with a non-zero status if any test suite fails; tracefiles
# are written regardless.

using CoverageTools
using Printf

const REPO_DIR = realpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(@__DIR__, "results")
const PACKAGES = ["SymRegInterpreter", "SymRegLikelihoods", "SymRegModelSelection"]

function parse_args(args)
    html = "--html" in args
    pkgs = filter(!=("--html"), args)
    for pkg in pkgs
        pkg in PACKAGES ||
            error("unknown package `$pkg`; expected one of $(join(PACKAGES, ", "))")
    end
    return (isempty(pkgs) ? PACKAGES : pkgs), html
end

# Runs the test suite of `pkg` in a fresh process, tracking coverage only for
# files in `srcdir`, and returns whether the tests passed.
function run_tests_with_coverage(pkg, srcdir)
    coverage = "@" * abspath(srcdir)
    code = "using Pkg; Pkg.instantiate(); Pkg.test(; coverage = $(repr(coverage)))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$pkg -e $code`
    return success(run(ignorestatus(cmd)))
end

function coverage_percent(fcs)
    covered, total = get_summary(fcs)
    return total == 0 ? 0.0 : 100 * covered / total, covered, total
end

function main(args)
    pkgs, html = parse_args(args)
    mkpath(RESULTS_DIR)
    failed = String[]
    summaries = Tuple{String, Float64, Int, Int}[]
    all_fcs = FileCoverage[]

    # relative paths keep the tracefiles independent of the checkout location
    cd(REPO_DIR) do
        for pkg in pkgs
            srcdir = joinpath(pkg, "src")
            @info "Testing $pkg with coverage"
            clean_folder(srcdir)
            run_tests_with_coverage(pkg, srcdir) || push!(failed, pkg)
            fcs = process_folder(srcdir)
            clean_folder(srcdir)

            LCOV.writefile(joinpath(RESULTS_DIR, "$pkg.info"), fcs)
            append!(all_fcs, fcs)
            push!(summaries, (pkg, coverage_percent(fcs)...))
        end
    end

    tracefile = joinpath(RESULTS_DIR, "lcov.info")
    LCOV.writefile(tracefile, all_fcs)

    println("\nLine coverage")
    for (pkg, pct, covered, total) in summaries
        @printf("  %-22s %6.2f%%  (%d / %d lines)\n", pkg, pct, covered, total)
    end
    pct, covered, total = coverage_percent(all_fcs)
    @printf("  %-22s %6.2f%%  (%d / %d lines)\n", "total", pct, covered, total)
    println("\nLCOV tracefile: ", relpath(tracefile, REPO_DIR))

    if html
        htmldir = joinpath(RESULTS_DIR, "html")
        if Sys.which("genhtml") === nothing
            @warn "`genhtml` not found; skipping HTML report (install lcov)"
        else
            cd(REPO_DIR) do
                run(`genhtml --quiet $tracefile --output-directory $htmldir`)
            end
            println("HTML report:    ", relpath(joinpath(htmldir, "index.html"), REPO_DIR))
        end
    end

    if !isempty(failed)
        @error "Test failures in: $(join(failed, ", "))"
        exit(1)
    end
    return nothing
end

main(ARGS)
