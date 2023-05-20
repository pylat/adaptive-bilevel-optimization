function download_maybe(url::AbstractString, local_path::AbstractString)
    filename = basename(url)
    full_path = joinpath(local_path, filename)
    if !isfile(full_path)
        mkpath(local_path)
        @info "Downloading $(filename) to $(local_path)"
        download(url, full_path)
        @info "Download finished"
    else
        @info "File $(filename) is already in $(local_path)"
    end
    return full_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    target_path = joinpath(@__DIR__, "experiments/datasets")
    download_maybe(
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a",
        target_path,
    )
    download_maybe(
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms",
        target_path,
    )
    download_maybe(
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
        target_path,
    )
end
