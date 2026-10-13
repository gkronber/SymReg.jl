# julia-downgrade-compat restores only `path` sources to the resolved manifest;
# `url` sources are dropped, so Pkg then fails with "depends on X, but no such
# entry exists in the manifest". This script clones every `url` source into
# `clone_dir` and rewrites its `[sources]` entry as a `path` source. Nested
# sources of path and url sources are rewritten as well.
#
# usage: julia url_sources_to_paths.jl <project dir> <clone dir>
using TOML

function rewrite_sources!(project_dir, clone_dir; cloned = Dict{Tuple{String, String}, String}(), visited = Set{String}())
    project_dir = normpath(project_dir)
    project_dir in visited && return
    push!(visited, project_dir)
    project_file = joinpath(project_dir, "Project.toml")
    project = TOML.parsefile(project_file)
    sources = get(project, "sources", Dict{String, Any}())
    changed = false
    for (name, source) in sources
        source isa AbstractDict || continue
        if haskey(source, "path")
            rewrite_sources!(joinpath(project_dir, source["path"]), clone_dir; cloned, visited)
            continue
        end
        haskey(source, "url") || continue
        url, rev = source["url"], get(source, "rev", "")
        repo_dir = get!(cloned, (url, rev)) do
            dir = joinpath(clone_dir, splitext(basename(url))[1] * (isempty(rev) ? "" : "-$rev"))
            branch = isempty(rev) ? `` : `--branch $rev`
            run(`git -c advice.detachedHead=false clone --quiet --depth 1 $branch $url $dir`)
            dir
        end
        pkg_dir = joinpath(repo_dir, get(source, "subdir", ""))
        sources[name] = Dict("path" => relpath(pkg_dir, project_dir))
        @info "Rewrote url source $name of $project_file as path source $(sources[name]["path"])"
        changed = true
        rewrite_sources!(pkg_dir, clone_dir; cloned, visited)
    end
    changed && open(io -> TOML.print(io, project; sorted = true), project_file, "w")
    return
end

rewrite_sources!(abspath(ARGS[1]), abspath(ARGS[2]))
