using SparseArrays
using LinearAlgebra

function get_finite_diff(m)
    I = vcat(1:m-1, 1:m-1)
    J = vcat(1:m-1, 2:m)
    V = vcat(ones(m-1), .-ones(m-1))
    return sparse(I, J, V) 
end

struct FiniteDiff2D{S, T}
    D_rows::S
    D_cols::T
end

struct AdjointOperator{O}
    op::O
end

function FiniteDiff2D(m::Int, n::Int)
    D_rows = get_finite_diff(m)
    @assert size(D_rows) == (m-1, m)
    D_cols = get_finite_diff(n)
    @assert size(D_cols) == (n-1, n)
    return FiniteDiff2D(D_rows, D_cols)
end

size_row_diffs(D::FiniteDiff2D) = (size(D.D_rows, 1), size(D.D_cols, 2))
size_col_diffs(D::FiniteDiff2D) = (size(D.D_rows, 2), size(D.D_cols, 1))

input_size(D::FiniteDiff2D) = (size(D.D_rows, 2), size(D.D_cols, 2))
output_size(D::FiniteDiff2D) = (prod(size_row_diffs(D)) + prod(size_col_diffs(D)),)

function Base.:*(D::FiniteDiff2D, x)
    O_rows = D.D_rows * x
    @assert size(O_rows) == size_row_diffs(D)
    O_cols = x * D.D_cols'
    @assert size(O_cols) == size_col_diffs(D)
    return vcat(vec(O_rows), vec(O_cols))
end

LinearAlgebra.adjoint(D::FiniteDiff2D) = AdjointOperator(D)

function Base.:*(D::AdjointOperator{<:FiniteDiff2D}, x)
    R = size_row_diffs(D.op)
    C = size_col_diffs(D.op)
    O_rows = reshape(x[1:prod(R)], R)
    O_cols = reshape(x[prod(R)+1:prod(R)+prod(C)], C)
    return (D.op.D_rows' * O_rows) + (O_cols * D.op.D_cols)
end

function test_operator(op; N=100)
    m, n = input_size(op)
    size_out = output_size(op)
    for _ in 1:N
        x = randn(m, n)
        row_diffs = x[1:m-1, :] - x[2:m, :]
        col_diffs = x[:, 1:n-1] - x[:, 2:n]
        @assert isapprox(op * x, vcat(vec(row_diffs), vec(col_diffs)))
        y = randn(size_out)
        @assert isapprox(dot(y, op * x), dot(op' * y, x))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    D = FiniteDiff2D(200, 300)
    test_operator(D)
end
