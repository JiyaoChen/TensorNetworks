using TensorKit
using TensorOperations
using KrylovKit

include("engines/DMRG_contractions.jl")

# clear console
Base.run(`clear`)

# no symmetry
d = 4  # physical dimension
D = 16  # MPO bond dimension
M = 64  # MPS bond dimension

𝕕 = ℂ^d  # physical link space
𝔻 = ℂ^D  # MPO link space
𝕄 = ℂ^M  # MPS link space

elemtype = Complex{Float64}

println("Creation of T")
@time T = TensorMap(randn,elemtype, 𝕄 ⊗ 𝕕 ⊗ 𝕕, 𝕄)  # the usual two-site MPS tensor
@time Tpr = TensorMap(convert(Array, T'), 𝕄 ⊗ 𝕕 ⊗ 𝕕, 𝕄)  # the usual two-site MPS tensor
@time T = T + Tpr  # make hermitian
@time norm = tr(T'*T)
@time T = T/sqrt(norm)

println("Creation of W")
@time W = TensorMap(randn,elemtype, 𝔻 ⊗ 𝕕, 𝔻 ⊗ 𝕕)  # the usual MPO tensor
@time W = W+W' # make hermitian
@time norm = tr(W*W')
@time W = W/sqrt(norm)

@time W1 = TensorMap(randn,elemtype, 𝔻 ⊗ 𝕕, 𝔻 ⊗ 𝕕)  # the usual MPO tensor
@time W2 = TensorMap(randn,elemtype, 𝔻 ⊗ 𝕕, 𝔻 ⊗ 𝕕)  # the usual MPO tensor

println("Creation of L")
@time L = TensorMap(ones,elemtype, 𝕄, 𝔻 ⊗ 𝕄)  # the corresponding left environment
println("Creation of R")
@time R = TensorMap(ones, 𝕄 ⊗ 𝔻, 𝕄)  # the corresponding right environment

# @time @tensor trash[:] := L[-1 2 1]*T[1 3 -2 -3]*W[2 -4 -5 3]

# applyH(T, L, W, W, R)

println("W*W")
@time W * W  # this is good
@time @tensor trash[:] := W[-1 -2 1 2] * W[1 2 -3 -4]  # this is bad

println("T'*T")
@time T' * T  # this is good
@time @tensor trash[:] := T[1 2 3 -4] * conj(T[1 2 3 -5])  # this is bad

# println("T*T'")
# @time T * T'  # this is good
# @time @tensor trash[:] := T[-1 -2 -3 1] * conj(T[-4 -5 -6 1])  # this is bad

# function calcTW(T,W)
#     trash = permute(T, (1,3,4), (2,)) * permute(W, (4,), (1,2,3))  # this is good
#     return trash
# end
# function calcTW2(T,W)
#     @tensor trash[:] := T[-1 1 -2 -3] * W[-4 -5 -6 1]  # this is bad
#     return trash
# end

# trash = calcTW(T,W)
# @time calcTW(T,W)
# @time calcTW(T,W)

# trash = calcTW2(T,W)
# @time calcTW2(T,W)
# @time calcTW2(T,W)

# function permuteLT(L,T)
#     trash = permute(L, (1,2), (3,)) * permute(T, (1,), (2,3,4))  # this is good
#     return trash
# end
# function tensorLT(L,T)
#     @tensor trash[:] := L[-1 -2 1] * T[1 -3 -4 -5]  # this is bad
#     return trash
# end

# LT = permuteLT(L,T)
# @time permuteLT(L,T)
# @time permuteLT(L,T)

# LT = tensorLT(L,T)
# @time tensorLT(L,T)
# @time tensorLT(L,T)


# function permuteLT_W(LT,W)
#     trash = permute(LT, (1,4,5), (2,3)) * permute(W, (1,4), (2,3))
#     return trash
# end
# function tensorLT_W(LT,W)
#     @tensoropt trash[:] := LT[-1 1 2 -2 -3] * W[1 -4 -5 2]
#     return trash
# end
# function tensorLTW(L,T,W)
#     @tensoropt trash[:] := L[-1 2 1] * T[1 3 -2 -3] * W[2 -4 -5 3]
#     return trash
# end

# LTW = permuteLT_W(LT,W)
# @time LTW = permuteLT_W(permuteLT(L,T),W)
# @time LTW = permuteLT_W(permuteLT(L,T),W)

# LTW = tensorLT_W(tensorLT(L,T),W)
# @time LTW = tensorLT_W(tensorLT(L,T),W)
# @time LTW = tensorLT_W(tensorLT(L,T),W)

# LTW = tensorLTW(L,T,W)
# @time LTW = tensorLTW(L,T,W)
# @time LTW = tensorLTW(L,T,W)

# function tensorLTW_W(LTW,W2)
#     @tensoropt trash[:] := LTW[-1 1 -2 -3 2] * W2[2 -4 -5 1]
#     return trash
# end
# function tensorLTWW(L,T,W1,W2)
#     @tensoropt trash[:] := L[-1 2 1] * T[1 3 5 -2] * W1[2 -3 4 3] * W2[4 -4 -5 5]
#     return trash
# end

function tensorLTWWR1(L,T,W1,W2,R)  # this is good :)
    # println("called LTWWR1")
    @tensor T[-1 -2 -3; -4] := L[-1 2 1] * T[1 3 5 6] * W1[2 -2 4 3] * W2[4 -3 7 5] * R[6 7 -4] order=(1,2,3,4,5,6,7)
    return T
end

function tensorLTWWR2(L,T,W1,W2,R)
    @tensor T[:] := L[-1 2 1] * T[1 3 5 6] * W1[2 -2 4 3] * W2[4 -3 7 5] * R[6 7 -4] order=(1,2,3,5,4,6,7)
    return T
end

function tensorLTWWR_opt(L,T,W1,W2,R)
    @tensoropt T[α β γ δ] := L[α j i] * T[i k m n] * W1[j β l k] * W2[l γ o m] * R[n o δ]
    return T
end

function tensorLTWWR_vec(x,L,T,W1,W2,R)  # this is good :)
    # println("called LTWWR1")
    X = TensorMap(x, codomain(T), domain(T))
    X = tensorLTWWR1(L,X,W1,W2,R)
    x = convert(Array,X)
    return x
end

println("L*T*W*W*R -- first ordering")
LTWWR = tensorLTWWR1(L,T,W,W,R)
for i = 1:10
    @time global LTWWR = tensorLTWWR1(L,LTWWR,W,W,R)
end

println("L*T*W*W*R -- second ordering")
LTWWR = tensorLTWWR2(L,T,W,W,R)
for i = 1:10
    @time global LTWWR = tensorLTWWR2(L,LTWWR,W,W,R)
end

println("L*T*W*W*R -- optimized???")
LTWWR = tensorLTWWR_opt(L,T,W,W,R)
for i = 1:10
    @time global LTWWR = tensorLTWWR_opt(L,LTWWR,W,W,R)
end

# tol = 1e-15
# maxiter = 12
# krylovdim = 10

# println("Arnoldi eigsolve")
# solver = Arnoldi
# for i = 1:3
#     @time eigenVal, eigenVec = eigsolve(x->tensorLTWWR1(L,x,W,W,R),T,1,:SR,solver(tol=tol,maxiter=maxiter,krylovdim=krylovdim))
#     println(abs(eigenVal[1]))
# end

# println("Lanczos eigsolve")
# solver = Lanczos
# for i = 1:3
#     @time eigenVal, eigenVec = eigsolve(x->tensorLTWWR1(L,x,W,W,R),T,1,:SR,solver(tol=tol,maxiter=maxiter,krylovdim=krylovdim))
#     println(abs(eigenVal[1]))
# end
# @time tensorLTWWR_vec(convert(Array, T), L, T, W1, W2, R)
# @time tensorLTWWR1(L, T, W1, W2, R)
# eigs(convert(Array, T), nev=1, which=:LR) do x tensorLTWWR_vec(x,L,T,W1,W2,R) end

0;  # to suppress output of the REPL