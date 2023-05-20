record_grad_count(_) = nothing
record_grad_count(c::Counting) = c.count_gradient

record_prox_count(_) = nothing
record_prox_count(c::Counting) = c.count_prox

record_mul_count(_) = nothing
record_mul_count(c::Counting) = c.count_mul

record_amul_count(_) = nothing
record_amul_count(c::Counting) = c.count_amul


record_backtracks(_) = 0.0
record_backtracks(c::Counting{Float64, Int64}) = c.count_mul

nocount(obj) = obj
nocount(c::Counting) = c.f

record_pg(x, f1, f2, g1, g2, gamma, sigma, norm_res, norm_gradf2, beta) = Dict(
    :objective1 => obj(f1,g1, x), 
    :objective2 => nocount(f2)(x), 
    :grad_f1_evals => record_grad_count(f1),
    :grad_f2_evals => record_grad_count(f2),
    :grad_evals_total => total_grad_count(f1, f2),
    :prox_g1_evals => record_prox_count(g1),
    :prox_g2_evals => record_prox_count(g2),
    :gamma => gamma,
    :backtracks => record_backtracks(beta),
    :sigma => sigma,
    :norm_res => norm_res,
    :norm_gradf2 => norm_gradf2,
)

function obj(f1, g1, x) 
    y = try 
        nocount(f1)(x) + nocount(g1)(x)
    catch e 
        nocount(f1)(x)
    end 
    return y
end



total_grad_count(f1::Counting, f2::Counting) =  record_grad_count(f1) + record_grad_count(f2)
total_grad_count(f1::Counting, f2) =  record_grad_count(f1) 
total_grad_count(f1, f2::Counting) =  record_grad_count(f2) 


concat_dicts(dicts) = Dict(k => [d[k] for d in dicts] for k in keys(dicts[1]))

function subsample(n, collection)
    step = length(collection) / n |> ceil |> Int
    return collection[1:step:end]
end

subsample(n) = collection -> subsample(n, collection)
