include(joinpath(@__DIR__, "../../", "counting.jl"))
include(joinpath(@__DIR__, "../../", "recording.jl"))
include(joinpath(@__DIR__, "../../", "adaptive_bilevel_algorithms.jl"))
include(joinpath(@__DIR__, "../../", "linops.jl"))

using Random
using LinearAlgebra
using MatrixDepot
using ProximalOperators: NormL1, IndBox, SqrNormL2, Quadratic

using DelimitedFiles
using Plots
using LaTeXStrings

pgfplotsx()


function run_Lin_inverse(functype; m = 20, r = 1e-2, seed = 0, tol = 1e-5, maxit = 10_000)
    @info "Start Linear inverse problem ($functype)"

    Random.seed!(seed)

    md = mdopen(functype, m, false)
    A = md.A
    bt = md.b
    xt = md.x
    n = size(A, 2)

    b = A * xt + r * randn(n)

    DisGrad = FiniteDiff2D((n, n)...)
    Q1 = DisGrad.D_rows' * DisGrad.D_rows
    Q = Q1 + diagm(0 => ones(n))

    Lf = opnorm(A)^2

    f1 = Quadratic(Q1, zeros(n))
    f1_BiGSAM = Quadratic(Q, zeros(n))
    f2 = LinearLeastSquares(A, b)
    g1 = SqrNormL2()
    g2 = IndBox(0, +Inf)

    # preparation for Big SAM
    muf1 = 1 # strong convexity modulus of f1
    Lf1 = opnorm(Q) # smoothness modulus
    Lf2 = Lf

    obj = obj1(f1, g1)

    @info "Getting accurate solution"

    sol_star, numit, record_fixed = adaptive_bilevel_LS(
        zeros(n),
        f1 = f1,
        f2 = Counting(f2),
        g1 = g1,
        g2 = g2,
        rule = OurRuleLS(gamma = 1.0),
        tol = tol,
        maxit = maxit * 50,
        record_fn = record_pg,
    )
    optimum = obj(sol_star)
    @info "high accuracy sol: $(optimum)"


    @info "Running solvers"

    @info "solver with bilevel problem with linesearch"

    sol, numit, record_Alg1LS = adaptive_bilevel_LS(
        zeros(n),
        f1 = f1,
        f2 = Counting(f2),
        g1 = g1,
        g2 = g2,
        rule = OurRuleLS(gamma = 10.0 / Lf2),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "Bilevel Alg with LS"
    @info "    iterations: $(numit)"
    @info "     objective: $(obj(sol))"


    @info "solver with bilevel problem with static stepsize"
    Lf1_staBiM = opnorm(Array(Q1))
    sol, numit, record_staBiM = adaptive_bilevel_static(
        zeros(n),
        f1 = f1,
        f2 = Counting(f2),
        g1 = g1,
        g2 = g2,
        rule = OurRule(sigma = 1.0, Lf = [Lf1_staBiM, Lf2]),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "Bilevel Alg with LS"
    @info "    iterations: $(numit)"
    @info "     objective: $(obj(sol))"


    @info "solver with bilevel problem with BiGSAM"

    sol, numit, record_BiGSAM = BiGSAM(
        zeros(n),
        f1 = f1_BiGSAM,
        f2 = Counting(f2),
        g = g2,
        gamma = 1 / Lf2,
        tau = 2 / (Lf1 + muf1),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "BiGSAM"
    @info "    iterations: $(numit)"
    @info "     objective: $(obj(sol))"


    @info "solver with bilevel problem with Solodov"

    record_Solodov = Vector{}(undef, 3)
    for (i, c) in [(1, 1), (2, 10), (3, 100)]
        sol, numit, record_Solodov[i] = backtracking_Solodov(
            zeros(n),
            f1 = f1_BiGSAM,
            f2 = Counting(f2),
            g = g2,
            gamma0 = c / Lf2,
            tol = tol,
            maxit = maxit,
            record_fn = record_pg,
        )
        @info "projprad"
        @info "    iterations: $(numit)"
        @info "     objective: $(obj(sol))"
    end
    @info "Collecting plot data"

    to_plot = Dict(
        "AdaBilevel-LS" => concat_dicts(record_Alg1LS |> subsample(100)),
        "staBiM" => concat_dicts(record_staBiM |> subsample(100)),
        "BiGSAM" => concat_dicts(record_BiGSAM |> subsample(100)),
        "Solodov1" => concat_dicts(record_Solodov[1] |> subsample(100)),
        "Solodov2" => concat_dicts(record_Solodov[2] |> subsample(100)),
        "Solodov3" => concat_dicts(record_Solodov[3] |> subsample(100)),
    )


    @info "Plotting"

    plot(
        title = "Quadratic upper level",
        xlabel = L"\nabla varphi_1\ \mbox{evaluations}",
        ylabel = L"\|v\|, v \in \partial \varphi_2(x^k)",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_evals_total],
            max.(1e-14, to_plot[k][:norm_res]),
            yaxis = :log,
            label = k,
        )
    end
    savefig(joinpath(@__DIR__, string("convergence_Linverse_res_", functype, "_.pdf")))

    plot(
        title = "Quadratic upper level",
        xlabel = L"\nabla varphi_1\ \mbox{evaluations}",
        ylabel = L"|f_1(x^k) - \varphi_{\star}|",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_evals_total],
            (abs.(to_plot[k][:objective1] .- optimum)) ./ max(1.0, optimum),
            yaxis = :log,
            label = k,
        )
    end
    savefig(joinpath(@__DIR__, string("convergence_Linverse_cost_", functype, "_.pdf")))

    plot(
        title = "Quadratic upper level",
        xlabel = L"\nabla varphi_1\ \mbox{evaluations}",
        ylabel = L"\gamma",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_evals_total],
            max.(1e-14, to_plot[k][:gamma]),
            yaxis = :log,
            label = k,
        )
    end
    savefig(joinpath(@__DIR__, string("convergence_Linverse_gamma_", functype, "_.pdf")))

    @info "Exporting plot data"

    save_labels = Dict(
        "AdaBilevel-LS" => "AdaBilevel-LS",
        "staBiM" => "staBiM",
        "BiGSAM" => "BiGSAM",
        "Solodov1" => "Solodov1",
        "Solodov2" => "Solodov2",
        "Solodov3" => "Solodov3",
    )

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_evals_total])
        rr = Int(ceil(d / 80)) # keeping at most 50 data points
        output =
            [to_plot[k][:grad_evals_total] (abs.(to_plot[k][:objective1] .- optimum)) ./
                                           max(1.0, optimum)]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$(functype)-$m-$n-$(Int(ceil(r*1000))).txt"
        filepath = joinpath(@__DIR__, "plotdata", "uppercost", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end


    for k in keys(to_plot)
        d = length(to_plot[k][:grad_evals_total])
        rr = Int(ceil(d / 80)) # keeping at most 50 data points
        output = [to_plot[k][:grad_evals_total] max.(1e-14, to_plot[k][:norm_res])]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$(functype)-$m-$n-$(Int(ceil(r*1000))).txt"
        filepath = joinpath(@__DIR__, "plotdata", "loweropt", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end

    for k in keys(to_plot)
        d = length(to_plot[k][:grad_evals_total])
        rr = Int(ceil(d / d)) # keeping at most 50 data points
        output = [1:d to_plot[k][:gamma]]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$(functype)-$m-$n-$(Int(ceil(r*1000))).txt"
        filepath = joinpath(@__DIR__, "plotdata", "gamma", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end

end


function main(; m = 1000)
    for r in [1e-2]
        run_Lin_inverse("phillips", m = m, r = r, maxit = 1000, tol = 1e-7)
        run_Lin_inverse("foxgood", m = m, r = r, maxit = 2000, tol = 1e-7)
        run_Lin_inverse("baart", m = m, r = r, maxit = 500, tol = 1e-7)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end



struct LinearLeastSquares{TA,Tb}
    A::TA
    b::Tb
end

(f::LinearLeastSquares)(w) = 0.5 * norm(f.A * w - f.b)^2

function gradient(f::LinearLeastSquares, w)
    res = f.A * w - f.b
    g = f.A' * res
    return g, 0.5 * norm(res)^2
end

function gradient(f::Quadratic, w)
    return f.Q * w + f.q, 0.5 * dot(w, f.Q * w) + dot(w, f.q)
end

function gradient(f::SqrNormL2, w)
    return w, 0.5 * norm(w)^2
end



struct obj1{Tf,Tg}
    f1::Tf                 # differentiable term  
    g1::Tg                 # nonsmooth term 
end


function (S::obj1)(x)
    y = try
        nocount(S.f1)(x) + nocount(S.g1)(x)
    catch e
        S.f1(x)
    end
    return y
end
