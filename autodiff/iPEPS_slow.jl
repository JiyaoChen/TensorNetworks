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
    # ctm = ctmtl(B, C, T)
    # ctm = permutedims(ctm, (2, 1, 4, 3))
    # CTM = reshape(ctm, (size(ctm,1)*size(ctm,2), size(ctm,3)*size(ctm,4)))
    # CTM += CTM'
    # U, Λ, V = svd(CTM)
    # Λ = Λ[1:size(C,1)]
    # Λ /= sqrt(sum(Λ.^2))
    # U = U[:, 1:size(C,1)]

    cp = ein"ad,iba,dcl,jkcb -> ijlk"(C, T, T, B)
    tp = ein"iam,jkla -> ijklm"(T, B)

    χ = size(C, 1)
    D = size(B, 1)

    # renormalize
    cpmat = reshape(cp, χ*D, χ*D)
    cpmat += adjoint(cpmat)
    u, s, v = svd(cpmat)

    z = reshape(u[:, 1:size(C,1)], (size(C,1), size(B,1), size(C,1)))

    C = ein"abcd,abi,cdj -> ij"(cp, conj(z), z)
    T = ein"abjcd,abi,dck -> ijk"(tp, conj(z), z)

    Λ = s[1:χ] ./ s[1]

    # indexperm_symmetrize
    C += C'
    T += ein"ijk -> kji"(conj(T))

    # normalize
    C /= norm(C)
    T /= norm(T)

    # Cpr = U'*CTM*U
    # # Cpr /= sqrt(tr(C*C'))
    # Cpr += Cpr'
    # Cpr /= norm(Cpr)

    # ctm = ctmcl(B, T)
    # CTM = reshape(ctm, (size(ctm,1)*size(ctm,2), size(ctm,3)*size(ctm,4)*size(ctm,5)))
    # Tpr = U'*CTM
    # Tpr = reshape(Tpr, (size(ctm,1)*size(ctm,3),size(ctm,4)*size(ctm,5)))
    # Tpr = Tpr*U
    # Tpr = reshape(Tpr, size(T))

    # Tpr += ein"ijk -> kji"(conj(Tpr))
    # Tpr /= norm(Tpr)

    return C, T, Λ
end

σ₀ = [1.0 0; 0  1.0]
σ₁ = [0 1.0; 1.0  0]
σ₃ = [1.0 0; 0 -1.0]
σ₂ = -1im*σ₃*σ₁

⊗(A, B) = kron(A, B)

function isingTBG(h, J)
    reshape(h*(σ₃⊗σ₀ + σ₀⊗σ₃) + J*σ₁⊗σ₁, (2,2,2,2))
    # reshape(kron(σ₀, σ₀), (2,2,2,2))
end

function heisenbergTBG()
    TBG = reshape(kron(σ₁, σ₁) + kron(σ₂, σ₂) + kron(σ₃, σ₃), 2, 2, 2, 2);
    TBG = ein"ijkl, ja, lb -> iakb"(TBG, σ₁, σ₁);
    TBG = permutedims(TBG, (1, 3, 2, 4));
    return real(TBG)
end

D = 3
d = 2
χ = 5

B = randn(D, D, d, D, D)
gate = isingTBG(1., 0.)
gate = isingTBG(0., -1.)
# gate = heisenbergTBG()

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

function expectationvalue(ap, C, T, gate)
    ap /= norm(ap)
    l = ein"(ab,ica),bde,cjfdlm,eg,gfk -> ijklm"(C,T,T,ap,C,T)
    e = ein"abcij,abckl,ikjl -> "(l,l,gate)[]
    n = ein"ijkaa,ijkbb -> "(l,l)[]
    # println(e)
    # println(n)
    return e/n
end

function indexperm_symmetrize(x)
    # B += permutedims(B, (5,1,3,2,4))
    # B += permutedims(B, (4,5,3,1,2))
    # B += permutedims(B, (2,4,3,5,1))
    x = permutedims(x, (1,2,4,5,3))
    x += permutedims(x, (1,4,3,2,5)) # left-right
    x += permutedims(x, (3,2,1,4,5)) # up-down
    x += permutedims(x, (2,1,4,3,5)) # diagonal
    x += permutedims(x, (4,3,2,1,5)) # rotation
    x = permutedims(x, (1,2,5,3,4))
    return x / norm(x)
end

function iPEPS(B, gate)
    B = indexperm_symmetrize(B)
    # B /= norm(B)
    Dsq = size(B, 1)^2
    BBd = reshape(ein"acieg, bdjfh -> abcdefghij"(B, conj(B)), (Dsq, Dsq, Dsq, Dsq, d, d))
    BBdtr = ein"abcdii -> abcd"(BBd)
    # nrm = norm(reshape(BBd, (Dsq^2, Dsq^2)))
    # B /= nrm
    # BBd /= nrm

    C = rand(χ, χ)
    C += C'
    T = rand(χ, Dsq, χ)
    T += ein"ijk -> kji"(conj(T))

    C, T = ctmrg(BBdtr, C, T, 100, 1e-10)

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
        Δ -> Zygote.gradient(f,Δ)[1], B, LBFGS(), inplace = false, Optim.Options(f_tol = 1e-6, show_trace = true))
end

@info res.minimum iPEPS(res.minimizer, gate)