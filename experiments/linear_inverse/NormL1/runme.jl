include(joinpath(@__DIR__, "../../../", "counting.jl"))
include(joinpath(@__DIR__, "../../../", "recording.jl"))
include(joinpath(@__DIR__, "../../../", "adaptive_bilevel_algorithms.jl"))
include(joinpath(@__DIR__, "../../../", "linops.jl"))

using Random
using LinearAlgebra

using Plots
using LaTeXStrings
using DelimitedFiles

using ProximalOperators: NormL1, SqrNormL2

pgfplotsx()


function run_Linverse_sim(; seed = 3, m = 400, n =  1000, pf = 5, r = 40, tol = 1e-5, maxit = 10_000)
    @info "Start Lasso"

    Random.seed!(seed)
    
    p = n / pf # nonzeros
    rho = 1 # some positive constant controlling how large solution is
    lam = 1  

    y_star = rand(m)
    y_star ./= norm(y_star) #y^\star
    C = rand(m, n) .* 2 .- 1

    CTy = abs.(C' * y_star)
    perm = sortperm(CTy, rev = true) # indices with decreasing order by abs

    alpha = zeros(n)
    for i = 1:n
        if i <= p
            alpha[perm[i]] = lam / CTy[perm[i]]
        else
            temp = CTy[perm[i]]
            if temp < 0.1 * lam
                alpha[perm[i]] = lam
            else
                alpha[perm[i]] = lam * rand() / temp
            end
        end
    end
    A = C * diagm(0 => alpha)   
    # generate the primal solution
    x_star = zeros(n)
    for i = 1:n
        if i <= p
            x_star[perm[i]] = rand() * rho / sqrt(p) * sign(dot(A[:, perm[i]], y_star))
        end
    end
    b = A * x_star + y_star


    Lf = opnorm(A)^2

    f1 = Zero()
    g1 = NormL1()
    f2 = LinearLeastSquares(A, b)
    g2 = Zero()
    
    obj = obj1(f1, g1)
    # preparation for Big SAM
    muf1 = 1 # strong convexity modulus of f1_BGSAM
    Lf2 = Lf 

@info "Getting accurate solution"

    sol_star, numit, record_fixed = adaptive_bilevel_LS(
        zeros(n),
        f1 = f1,
        f2 = Counting(f2),
        g1 = g1,
        g2 = g2,
        rule = OurRuleLS(gamma = 1.0),
        tol = tol,
        maxit = maxit * 20,
        record_fn = record_pg,
    )
    optimum = obj(sol_star) 
    @info "high accuracy sol: $(optimum)"


    @info "Running solvers" 

    @info "solver with bilevel problem with LS"

    sol, numit, record_Alg1LS = adaptive_bilevel_LS(
        zeros(n),
        f1 = f1,
        f2 = Counting(f2),
        g1 = g1,
        g2 = g2,
        rule = OurRuleLS(gamma = 1.0),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "Bilevel Alg with LS"
    @info "    iterations: $(numit)"
    @info "     objective: $(obj(sol))"


@info "solver with bilevel problem with static stepsize"

    sol, numit, record_staBiM = adaptive_bilevel_static(
        zeros(n),
        f1 = f1,
        f2 = Counting(f2),
        g1 = g1,
        g2 = g2,
        rule = OurRule(sigma = 1.0, Lf = [0.0, Lf2]),
        tol = tol,
        maxit = maxit,
        record_fn = record_pg,
    )
    @info "Bilevel Alg with LS"
    @info "    iterations: $(numit)"
    @info "     objective: $(obj(sol))"


    @info "Collecting plot data"

    to_plot = Dict(
        "AdaBilevel-LS" => concat_dicts(record_Alg1LS |> subsample(100)),
        "staBiM" => concat_dicts(record_staBiM |> subsample(100)),
        )

    @info "Plotting"

    plot(
        title = "Quadratic upper level",
        xlabel = L"\nabla f_1\ \mbox{evaluations}",
        ylabel = L"\|v\|, v \in \partial \varphi_2(x^k)",
    )
    for k in keys(to_plot)
        plot!(
            to_plot[k][:grad_evals_total],
            max.(1e-14, to_plot[k][:norm_gradf2]),
            yaxis = :log,
            label = k,
        )
    end
    savefig(string("convergence_Linverse_res",".pdf"))

    plot(
        title = "Quadratic upper level",
        xlabel = L"\nabla f_2\ \mbox{evaluations}",
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
    savefig(string("convergence_Linverse_cost", ".pdf"))

    plot(
        title = "Quadratic upper level",
        xlabel = L"\nabla f_2\ \mbox{evaluations}",
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
    savefig(string("convergence_Linverse_gamma", ".pdf"))


    r = pf

    @info "Exporting plot data"

    save_labels = Dict(
        "AdaBilevel-LS" => "AdaBilevel-LS",
        "staBiM" => "staBiM",
    )


    for k in keys(to_plot)
        d = length(to_plot[k][:grad_evals_total])
        rr = Int(ceil(d / 80)) # keeping at most 50 data points
        output = [to_plot[k][:grad_evals_total] (abs.(to_plot[k][:objective1] .- optimum)) ./ max(1.0, optimum) ]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$m-$n-$(r)-$(pf).txt"
        filepath = joinpath(@__DIR__,"plotdata", "uppercost", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end


    for k in keys(to_plot)
        d = length(to_plot[k][:grad_evals_total])
        rr = Int(ceil(d / 80)) # keeping at most 50 data points
        output = [to_plot[k][:grad_evals_total] max.(1e-14, to_plot[k][:norm_gradf2])]
        red_output = output[1:rr:end, :]
        filename = "$(save_labels[k])-$m-$n-$(r)-$(pf).txt"
        filepath = joinpath(@__DIR__,"plotdata", "loweropt", filename)
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
        filename = "$(save_labels[k])-$m-$n-$(r)-$(pf).txt"
        filepath = joinpath(@__DIR__,"plotdata", "gamma", filename)
        mkpath(dirname(filepath))
        open(filepath, "w") do io
            writedlm(io, red_output)
        end
    end


end



function main(; maxit = 10_000)
    col = [
        (400, 1000, 10),
        (400, 4000, 40),
        (400, 10000, 100),
    ]
    for (m, n, pf) in col
        run_Linverse_sim(
            m = m, n = n, pf = pf, 
            maxit = maxit, tol = 1e-7
        )
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