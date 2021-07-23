using Zygote
# using BackwardsLinalg
# using TensorOperations
using LinearAlgebra
using OMEinsum
using Optim

function ctmtl(B, C, T)
    res = ein"ad, dbc -> abc"(C, T)
    res = ein"ecd, abe -> abcd"(res, T)
    res = ein"aefd, ebcf -> abdc"(res, B)
    return res
end

function ctmcl(B, T)
    res = ein"afe, fbcd -> abced"(T, B)
    return res
end

function ctmstep(B, C, T)
    ctm = ctmtl(B, C, T)
    CTM = reshape(ctm, (size(ctm,1)*size(ctm,2), size(ctm,3)*size(ctm,4)))
    CTM += CTM'
    U, Λ, V = svd(CTM)
    Λ = Λ[1:size(C,1)]
    Λ /= sum(Λ.^2)
    U = U[:, 1:size(C,1)]
    Cpr = U'*CTM*U
    # Cpr /= sqrt(tr(C*C'))
    Cpr += Cpr'
    Cpr /= norm(Cpr)

    ctm = ctmcl(B, T)
    CTM = reshape(ctm, (size(ctm,1)*size(ctm,2), size(ctm,3)*size(ctm,4)*size(ctm,5)))
    Tpr = U'*CTM
    Tpr = reshape(Tpr, (size(Tpr,1)*size(ctm,3),size(ctm,4)*size(ctm,5)))
    Tpr = Tpr*U
    Tpr = reshape(Tpr, size(T))

    Tpr += ein"ijk -> kji"(conj(Tpr))
    # Tpr /= norm(Tpr)

    return Cpr, Tpr, Λ
end

σ₀ = [1 0; 0  1]
σ₁ = [0 1; 1  0]
σ₃ = [1 0; 0 -1]
σ₂ = -1im*σ₃*σ₁

⊗(A, B) = kron(A, B)

function isingTBG(h, J)
    reshape(h*(σ₃⊗σ₀ + σ₀⊗σ₃) + J*σ₁⊗σ₁, (2,2,2,2))
end

D = 3
d = 2
χ = 5

B = randn(D, D, d, D, D)
gate = isingTBG(0., 1.)

function ctmrg(B, C, T, maxit, tol)
    oldvals = fill(Inf, size(C, 1))
    ctr = 1
    while ctr < maxit
        C, T, Λ = ctmstep(B, C, T)
        ϵ = sum(sqrt.(abs.(Λ .- oldvals).^2))
        if tol > ϵ
            return C, T 
        end
        oldvals = Λ
        ctr += 1
    end
    return C, T 
end

function energy(B, C, T, gate)
    lh = ein"ab, bdce, fghic, jkfa, klnmd, oj, plgo -> pmnhie"(C, T, B, T, conj(B), C, T)
    ene = ein"abcdef, cghd, abcdef ->"(lh, gate, lh)
    return ene[1]
end

function expectationvalue(ap, C, T, gate)
    ap /= norm(ap)
    l = ein"ab,ica,bde,cjfdlm,eg,gfk -> ijklm"(C,T,T,ap,C,T)
    e = ein"abcij,abckl,ijkl -> "(l,l,gate)[]
    n = ein"ijkaa,ijkbb -> "(l,l)[]
    return e/n
end

function indexperm_symmetrize(B)
    B += permutedims(B, (5,1,3,2,4))
    B += permutedims(B, (4,5,3,1,2))
    B += permutedims(B, (2,4,3,5,1))
    return B / norm(B)
end

function iPEPS(B, gate)
    B = indexperm_symmetrize(B)
    Dsq = size(B, 1)^2
    BBd = reshape(ein"acieg, bdjfh -> abcdefghij"(B, conj(B)), (Dsq, Dsq, Dsq, Dsq, d, d))
    BBdtr = ein"abcdii -> abcd"(BBd)
    # nrm = norm(reshape(BBd, (Dsq^2, Dsq^2)))
    # B /= nrm
    # BBd /= nrm

    C = rand(χ, χ)
    C += C'
    T = rand(χ, Dsq, χ)
    T += permutedims(T, [3, 2, 1])

    C, T = ctmrg(BBdtr, C, T, 50, 1e-12)

    # nrm = norm2(BBdtr, C, T)
    # T = reshape(T, (χ, D, D, χ))
    # nrg = energy(B, C, T, gate)
    nrg = expectationvalue(BBd, C, T, gate)

    println(nrg)

    return nrg
end

@info iPEPS(B, gate)

res=Nothing
let f = B -> iPEPS(B, gate)
    global res = optimize(f,
        Δ -> Zygote.gradient(f,Δ)[1], B, LBFGS(m = 20), inplace = false)
end

@info res.minimum iPEPS(res.minimizer, gate)