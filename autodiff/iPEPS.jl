using Zygote
# using BackwardsLinalg
using TensorOperations
using LinearAlgebra
using OMEinsum
using Optim

function CTMtl(bulk, C, T)
    ctmtl = ein"ab, bdce, fgha, hijmc, gkjld -> fikeml"(C, T, T, bulk, conj(bulk))
    return ctmtl
end

function CTMcl(bulk, T)
    # ctmcl = ein"abcd, cefgh, bifjk -> aeijgdhk"(T, bulk, conj(bulk))
    # ctmcl = ein"abcd, cefgh, bifjk -> aeijgkhd"(T, bulk, conj(bulk))
    ctmcl = ein"abcd, cefgh, bifjk -> ieajgdhk"(T, bulk, conj(bulk))
    return ctmcl
end

function ctmstep(bulk, T, C, χ)
    ctm = CTMtl(bulk, C, T)
    ctm = reshape(ctm, (χ*D^2, χ*D^2))
    ctm = 0.5.*(ctm + ctm')
    ctm /= tr(ctm*ctm')

    U, Λ, V = svd(ctm)
    Vdag = V';

    Λ /= sum(Λ.^2)

    U = U[:,1:χ]
    Δρ = sum(Λ[χ+1:end].^2)
    Λ = Λ[1:χ]
    Vdag = Vdag[1:χ,:]

    Vdag = U'
    C = U' * ctm * Vdag'
    C /= tr(C*C')

    etm = CTMcl(bulk, T)
    T = reshape(U'*reshape(etm, (χ*D^2,χ*D^4)), (χ*D^2,χ*D^2))*U
    T /= tr(T*T')
    T = reshape(T, (χ, D, D, χ))

    return T, C, Λ
end

function ctmrg(C, T, bulk, χ; tol::Real, maxit::Integer)
    # initialize
    oldvals = fill(Inf, χ)

    counter = 0
    Λ = Nothing
    while counter<maxit
        T, C, Λ = ctmstep(bulk, T, C, χ)
        # println(size(Λ))
        # println(size(oldvals))
        diff = sum(sqrt.(abs.(Λ .- oldvals).^2))
        # @info diff, counter
        diff <= tol && return T, C, Λ
        counter += 1
        oldvals = Λ
    end
    return T, C, Λ
end

# ctmrg(C, T, bulk, χ, tol=1e-6, maxit=100)

function energy(C, T, B, gate)
    lh = ein"ab, bdce, fghic, jkfa, klnmd, oj, plgo -> pmnhie"(C, T, B, T, conj(B), C, T)
    ene = ein"abcdef, cghd, abcdef ->"(lh, gate, lh)
    return ene[1]
end

function norm(C, T, B)
    # n = ein"ab, bdce, fghic, jkfa, klhmn, pj, olgp, qo, rmiq ->"(C, T, B, T, conj(B), C, T, C, T)
    n = @ein"ed, abce -> abcd"(C, T)  # O(χ³D²)
    n = @ein"ahfg, ehdbc -> abcdefg"(n, conj(B))  # O(χ²D⁵)
    n = @ein"abcdefg, ehdbc -> abcdefg"(n, T)  # O(χ³D⁵d)
    return n[1]
end

σ₀ = [1 0; 0  1]
σ₁ = [0 1; 1  0]
σ₃ = [1 0; 0 -1]
σ₂ = -1im*σ₃*σ₁

⊗(A, B) = kron(A, B)

function isingTBG(h, J)
    reshape(h*(σ₃⊗σ₀ + σ₀⊗σ₃) + J*σ₁⊗σ₁, (2,2,2,2))
end

D = 2
d = 2
χ = 2
bulk = randn(D, D, d, D, D)
gate = isingTBG(1., 0.)
function iPEPS(bulk, gate, χ, D)
    T = randn(χ, D, D, χ)
    C = randn(χ, χ)
    
    T, C, Λ = ctmrg(C, T, bulk, χ; tol=1e-12, maxit=5)
    ene = energy(C,T,bulk,gate) 
    # norm = norm(C,T,bulk)
    return ene
end

for ii=1:10
    println(iPEPS(bulk, gate, χ, D))
end

res=Nothing
let energy = bulk -> iPEPS(bulk, gate, χ, D)
    global res = optimize(energy,
        Δ -> Zygote.gradient(energy,Δ)[1], bulk, LBFGS(m = 20), inplace = false)
end