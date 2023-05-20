using Logging
using LinearAlgebra
using ProximalCore: prox, Zero 
import ProximalCore: prox, prox!

# Utilities.

is_logstep(n; base = 10) = mod(n, base^(log(base, n) |> floor)) == 0

nan_to_zero(v) = ifelse(isnan(v), zero(v), v)


# Adaptive proximal bilevel algorithms.
# All algorithms implemented as special cases of one generic loop.


Base.@kwdef struct OurRuleLS{R}
    gamma::R
    sigk1::R
    gamax::R  
    beta::Union{R, Counting{R}}
end

function OurRuleLS(;gamma = 0, sigk1 = 1.0, gamax = one(gamma)*1000_000, beta = Counting(0.5))
    _gamma = if gamma > 0
        gamma
    else
        error("you must provide gamk1 > 0")
    end
    _sigk1 = if sigk1 > 0
        sigk1
    else
        error("you must provide sigma > 0")
    end
    R = typeof(gamma)
    return OurRuleLS{R}(_gamma, _sigk1, gamax, beta)
end

function stepsize(rule::OurRuleLS)
    gamma = rule.gamma
    sigk1 = rule.sigk1
    R = typeof(gamma)
    return gamma, sigk1
end


# AdaBiM algorithm for convex structured bilevel problems 
#
# See Latafat, P., Themelis, A., Villa, S., and Patrinos, P. (2023). AdaBiM: An adaptive proximal gradient method 
# for structured convex bilevel optimization. arXiv preprint arXiv:2305.03559.

function adaptive_bilevel_LS(
    x;
    f1,
    f2,
    g1,
    g2, 
    rule,
    mu = 0.0,
    nu = 0.98,
    tol = 1e-5,
    maxit = 10_000,
    record_fn = nothing,
    name = "AdaBiM",
)
    gamk1, sigk1 = stepsize(rule)
    beta = rule.beta #backtrack param

    record = []
    x_prev = x
    grad1_x_prev, _ = gradient(f1, x_prev)
    grad2_x_prev, _ = gradient(f2, x_prev)
    grad_x_prev = sigk1 * grad1_x_prev + grad2_x_prev

    x, _ = prox(g1, g2, x_prev - gamk1 * grad_x_prev, gamk1, sigk1) # x0

    grad1_x, _ = gradient(f1, x)
    grad2_x, _ = gradient(f2, x)
    grad_x = sigk1 * grad1_x + grad2_x
    gam_prev  = gamk1
    sig_prev = sigk1
    sigk = sigk1
    for it = 1:maxit
        C = norm(grad_x - grad_x_prev)^2 / dot(grad_x - grad_x_prev, x - x_prev) |> nan_to_zero
        L = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2 |> nan_to_zero
        L2 = dot(grad2_x - grad2_x_prev, x - x_prev) / norm(x - x_prev)^2 |> nan_to_zero
        D = gamk1 * L * (gamk1 * C - 1) - gamk1*sigk * mu /2 |> nan_to_zero
        
        tau_prev = sigk/sig_prev
        sigk1 = min(max(1/it, 3* sigk / 4), sigk) # sig+  

        tau =  sigk1/sigk

        rho = gamk1 / gam_prev * tau_prev
        gam_prev = gamk1
        gamk1 = min(
            (gam_prev / tau) * sqrt( (1 + rho)*tau_prev ),
            (gam_prev / tau) * sqrt( 1 + 2 * mu * sigk * gam_prev - 4*( 1- tau_prev ) * gam_prev * L2) / (2* sqrt(max(D,0))),
            rule.gamax,
        )

        sigk, sig_prev = sigk1, sigk

        grad_x = sigk1 * grad1_x + grad2_x # nabla f_{k+1}(x^k)
        
        grad2_x_prev = grad2_x 
        grad1_x_prev = grad1_x 
        x_prev, grad_x_prev = x, grad_x
        while true
            v = x_prev - gamk1 * grad_x_prev
            x, _ = prox(g1, g2, v, gamk1, sigk1)
            grad1_x, _ = gradient(f1, x)
            grad2_x, _ = gradient(f2, x)
            grad_x = sigk1 * grad1_x + grad2_x

            ell = dot(grad_x - grad_x_prev, x - x_prev) / norm(x - x_prev)^2 |> nan_to_zero
            if gamk1 * (ell - sigk * mu / 2 ) <= nu + 1e-12
                break
            end
            gamk1 = beta * gamk1
            if gamk1 <= 1e-8
                @info("stepsize in the linesearch is too small")
            end
        end
        norm_res = norm_res_eval(g1, x_prev, x, gamk1, sigk1, grad2_x, grad_x_prev)

        if record_fn !== nothing
            push!(record, record_fn(x, f1, f2, g1, g2, gamk1, sigk1, norm_res, norm(grad2_x), beta) )   ##### bug prone
        end
        if is_logstep(it, base = 10)
            @info "$name" it norm_res
        end
    end
    return x, maxit, record
end

function norm_res_eval(g1::Zero, x_prev, x, gamk1, sigk1, grad2_x, grad_x_prev)
    norm_res = norm( (x_prev - x)./gamk1 + grad2_x - grad_x_prev)
    return norm_res
end 

function norm_res_eval(g1, x_prev, x, gamk1, sigk1, grad2_x, grad_x_prev)
    norm_res = norm( (x_prev - x)./gamk1 + grad2_x - grad_x_prev)
    return norm_res
end 

function norm_res_eval(g1::SqrNormL2, x_prev, x, gamk1, sigk1, grad2_x, grad_x_prev)
    norm_res = norm( (x_prev - x)./gamk1 + grad2_x - grad_x_prev - sigk1 * x)
    return norm_res
end 


Base.@kwdef struct OurRule{R}
    sigk1::R
    Lf1::R 
    Lf2::R 
end

function OurRule(;sigma = 1.0, Lf = [0.0, 0.0])
    _sigk1 = if sigma > 0
        sigma
    else
        error("you must provide sigma > 0")
    end
    R = typeof(sigma)
    return OurRule{R}(_sigk1, Lf[1], Lf[2])
end

stepsize(rule::OurRule) = (rule.sigk1, rule.Lf1, rule.Lf2)


# StaBiM algorithm for convex structured bilevel problems 
#
# See Latafat, P., Themelis, A., Villa, S., and Patrinos, P. (2023). AdaBiM: An adaptive proximal gradient method 
# for structured convex bilevel optimization. arXiv preprint arXiv:2305.03559.


function adaptive_bilevel_static(
    x;
    f1,
    f2,
    g1,
    g2, 
    rule,
    tol = 1e-5,
    maxit = 10_000,
    record_fn = nothing,
    name = "StaBiM",
)
    sigk1, Lf1, Lf2 = stepsize(rule)
    gamk1 = 0.99 / (sigk1 * Lf1 + Lf2)

    record = []
    grad1_x, _ = gradient(f1, x)
    grad2_x, _ = gradient(f2, x)
    grad_x = sigk1 * grad1_x + grad2_x
    v = x - gamk1 * grad_x
    x, _ = prox(g1, g2, v, gamk1, sigk1)

    for it = 1:maxit
        sigk1 = min(max(1/it, 3* sigk1 / 4), sigk1)
        gamk1 = 0.99 / (sigk1 * Lf1 + Lf2)
    
        x_prev= x
        grad_x_prev = sigk1 * grad1_x + grad2_x

        grad1_x, _ = gradient(f1, x)
        grad2_x, _ = gradient(f2, x)
        grad_x = sigk1 * grad1_x + grad2_x

        v = x_prev - gamk1 * grad_x_prev
        x, _ = prox(g1, g2, v, gamk1, sigk1)

        norm_res = norm_res_eval(g1, x_prev, x, gamk1, sigk1, grad2_x, grad_x_prev)

        if record_fn !== nothing
            push!(record, record_fn(x, f1, f2, g1, g2, gamk1, sigk1, norm_res, norm(grad2_x), 0.0) )   ##### bug prone
        end
        if is_logstep(it, base = 10)
            @info "$name" it norm_res
        end
    end
    return x, maxit, record
end




# SEDM: Solodov's explicit descent method 
#
# See Mikhail Solodov. An explicit descent method for bilevel convex optimization. Journal of Convex Analysis,
# 14(2):227, 2007.
 

function backtracking_Solodov(x0; f1, f2, g, gamma0, sigma0 =1.0, theta = 0.98, beta = Counting(0.5), tol = 1e-5, maxit = 100_000, record_fn = nothing)
    x, z, gamma, sigma = x0, x0, gamma0, sigma0
    grad1_x, f1_val = gradient(f1, x)
    grad2_x, f2_val = gradient(f2, x)
    grad_x = sigma * grad1_x + grad2_x
    f_x =  f1_val * sigma + f2_val

    record = []
    for it = 1:maxit
        gamma = gamma0
        z, _ = prox(g, x - gamma * grad_x, gamma)
        phik_x = f_x + g(x)
        ub_z = theta * real(dot(grad_x, z - x))
        while sigma * f1(z) + f2(z) + g(z)  > phik_x + ub_z
            gamma = beta * gamma
            if gamma < 1e-12
                @error "step size became too small ($gamma)"
            end
            z, _ = prox(g, x - gamma * grad_x, gamma)
            ub_z = theta * real(dot(grad_x, z - x))
        end
        grad1_z, f1_z = gradient(f1, z)
        grad2_z, f2_z = gradient(f2, z)
    

        sigma = min(sigma, 1/it)       
        grad_z = sigma * grad1_z + grad2_z
        f_z = sigma * f1_z + f2_z

        norm_res = norm( (x - z) ./ gamma + grad2_z - grad_x )
        if record_fn !== nothing
            push!(record, record_fn(x, f1, f2, Zero(), g, gamma, sigma, norm_res, norm(grad2_z), beta))  
        end
        if is_logstep(it, base = 10)
            @info "Backtracking PG with Armijo LS" it norm_res
        end
        x, f_x, grad_x = z, f_z, grad_z
    end
    return z, maxit, record
end

# BiG-SAM algorithm for convex bilevel problems 
#
# See Shoham Sabach and Shimrit Shtern. A first order method for solving convex bilevel optimization problems. SIAM
# Journal on Optimization, 27(2):640–660, 2017.


function BiGSAM(
    x;
    f1,
    f2,
    g, 
    gamma, # t
    tau, # s
    sigma = 1.0,
    tol = 1e-5,
    maxit = 1_000,
    record_fn = nothing,
    name = "BiGSAM",
)
    record = []
     for it = 1:maxit
        grad2_x, _ = gradient(f2, x)
        y, ~ = prox(g, x - gamma * grad2_x, gamma)
        grad1_x, _ = gradient(f1, x)
        z = x - tau * grad1_x
        x_prev = x
        x = sigma * z + (1 - sigma) * y

        sigma = sigma/ (1+ sigma)
        
        # optimality of lower level
        gradf2_x_temp, ~ = gradient(nocount(f2), y)
        norm_res = norm((x_prev - y) ./ gamma + gradf2_x_temp - grad2_x) # lower level
        if record_fn !== nothing
            push!(record, record_fn(x, f1, f2, Zero(), g, gamma, sigma, norm_res, norm(grad2_x), 0.0) ) 
        end
        if is_logstep(it, base = 10)
            @info "$name" it norm_res
        end
    end
    return x, maxit, record
end


#  Diagonal Dual Descent (3-D) algorithm for convex bilevel problems (implemented only for D_y = 1/2|. - y|^2 and R = 1/2|.|^2)
#
# See Guillaume Garrigos, Lorenzo Rosasco, and Silvia Villa. Iterative regularization via dual diagonal descent. Jour-
# nal of Mathematical Imaging and Vision, 60:189–215, 2018.



function Iterative3_D(
    x;
    f1,
    f2, 
    gamma, # tau
    sigma = 1.0, # alpha
    tol = 1e-5,
    maxit = 10_000,
    record_fn = nothing,
    name = "Iterative3D",
)
    record = []
    
    if norm(x) >= 1e-8
        @warn "Make sure x0 lies in the image of A'" 
    end

    
    for it = 1:maxit
        grad1_x, _ = gradient(f1, x)
        grad2_x, _ = gradient(f2, x)
        grad_x = sigma .* grad1_x + grad2_x
        x =  x  - gamma  .*  grad_x

        sigma = 1/it^2
        
        norm_res = norm(grad2_x)
        if record_fn !== nothing
            push!(record, record_fn(x, f1, f2, Zero(), Zero(), gamma, sigma, norm_res, norm(grad2_x), 0.0)) 
        end
        if is_logstep(it, base = 10)
            @info "$name" it norm_res
        end
    end
    return x, maxit, record
end



# evaluation of prox_{sigk g1 + g2}


function prox(g1::Zero, g2, x, gamma, sigk1)
    y = similar(x)
    gy = prox!(y, g2, x, gamma)
    return y, g1(y) * sigk1 + gy
end

function prox(g1, g2::Zero, x, gamma, sigk1)
    y = similar(x)
    gy = prox!(y, g1, x, gamma * sigk1)
    return y, gy * sigk1 + g2(y)
end

function prox(g1::Zero, g2::Zero, x, gamma, sigk1)
    return x, eltype(x)(0)
end


function prox(g1::SqrNormL2, g2, x, gamma, sigk1)
    R = real(eltype(x))
    y = similar(x)    

    mu = g1.lambda * sigk1
    gam = gamma
    
    gam_new = R(1) / (R(1) + mu * gam)
    gy = prox!(y, g2, gam_new .* x, gam_new .* gam)
    return y, sigk1* g1(y) + gy 
end


function prox(g1, g2::SqrNormL2, x, gamma, sigk1)
    R = real(eltype(x))
    y = similar(x)    

    gam = gamma*sigk1
    mu = g2.lambda * gamma
    gam_new = R(1) / (R(1) + mu * gam)
    gy = prox!(y, g1, gam_new .* x, gam_new .* gam)
    return y, sigk1 * gy + g2(y)
end

function prox(g1::Zero, g2::SqrNormL2, x, gamma, sigk1)
    y = similar(x)
    gy = prox!(y, g2, x, gamma)
    return y, gy
end


function prox(g1::SqrNormL2, g2::Zero, x, gamma, sigk1)
    y = similar(x)
    gy = prox!(y, g1, x, gamma * sigk1)
    return y, sigk1 * gy
end

function gradient(f::Zero, x)
    y = similar(x)
    y .= eltype(x)(0)
    return y, f(x)
end

