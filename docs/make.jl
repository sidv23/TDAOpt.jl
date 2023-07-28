using TDAOpt
using Documenter

DocMeta.setdocmeta!(TDAOpt, :DocTestSetup, :(using TDAOpt); recursive=true)

makedocs(;
    modules=[TDAOpt],
    authors="SidV <sidv@Lazuli.local> and contributors",
    repo="https://github.com/sidv23/TDAOpt.jl/blob/{commit}{path}#{line}",
    sitename="TDAOpt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sidv23.github.io/TDAOpt.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sidv23/TDAOpt.jl",
    devbranch="main",
)
