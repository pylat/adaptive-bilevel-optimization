using LinearAlgebra
using ProximalCore
using ProximalCore: prox, Zero
using ProximalOperators: SqrNormL2
import ProximalCore: prox, prox!

mutable struct Counting{F,I}
    f::F
    count_eval::I
    count_prox::I
    count_gradient::I
    count_mul::I
    count_amul::I
end

ProximalCore.is_convex(::Type{<:Counting{F}}) where {F} = ProximalCore.is_convex(F)
ProximalCore.is_generalized_quadratic(::Type{<:Counting{F}}) where {F} =
    ProximalCore.is_generalized_quadratic(F)

Counting(f::F) where {F} = begin
    count = 0
    Counting{F,typeof(count)}(f, count, count, count, count, count)
end

function (g::Counting)(x)
    g.count_eval += 1
    g.f(x)
end

### gradient counts

function gradient(g::Counting, x)
    g.count_gradient += 1
    gradient(g.f, x)
end

### prox count

function ProximalCore.prox!(y, g::Counting, x, gamma)
    g.count_prox += 1
    prox!(y, g.f, x, gamma)
end

function prox(g1::Zero, g2::Counting, x, gamma, sigma)
    y = similar(x)
    g2.count_prox += 1
    gy = prox!(y, g2, x, gamma)
    return y, gy
end

function prox(g1::Counting, g2::Zero, x, gamma, sigma)
    y = similar(x)
    g1.count_prox += 1
    gy = prox!(y, g1, x, gamma * sigma)
    return y, sigma * gy
end

function prox(g1::SqrNormL2, g2::Counting, x, gamma, sigma)
    R = real(eltype(x))
    y = similar(x)    
    g2.count_prox += 1

    gr = g1.lambda * gamma * sigma
    gr2 = R(1) / (R(1) + gr)
    gy = prox!(y, g2, gr2 .* x, gr2 .* gamma)

    return y, sigma*g1(y) + gy
end


function prox(g1::Counting, g2::SqrNormL2, x, gamma, sigma)
    R = real(eltype(x))
    y = similar(x)    
    g1.count_prox += 1

    gam = gamma*sigma
    mu = g2.lambda * gam
    gam_new = R(1) / (R(1) + mu * gam)
    gy = prox!(y, g1, gam_new .* x, gam_new .* gam)
    return y, sigma * gy + g2(y)
end


struct AdjointOperator{O}
    op::O
end

LinearAlgebra.norm(C::Counting) = norm(C.f)
LinearAlgebra.adjoint(C::Counting) = AdjointOperator(C)

function Base.:*(C::Counting, x)
    C.count_mul += 1
    return C.f * x
end

function Base.:*(A::AdjointOperator{<:Counting}, x)
    A.op.count_amul += 1
    return A.op.f' * x
end


# backtrack counting
function Base.:*(C::Counting{Float64, Int64}, x)
    C.count_mul += 1
    return C.f * x
end
