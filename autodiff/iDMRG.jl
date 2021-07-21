using Zygote
# using BackwardsLinalg
# using TensorOperations
using NCon
using LinearAlgebra
using OMEinsum


σ₀ = [1 0; 0  1]
σ₁ = [0 1; 1  0]
σ₃ = [1 0; 0 -1]
σ₂ = -1im*σ₃*σ₁

⊗(A, B) = kron(A, B)

function isingTBG(h, J)
    reshape(h*(σ₃⊗σ₀ + σ₀⊗σ₃) + J*σ₁⊗σ₁, (2,2,2,2))
end

function energy(x,gate)
    return @tensor x[1 2 3 4]*gate[5 6 2 3]*conj(x[1 5 6 4])
end

function energy2(x,gate)
    e = ein"abcd, efbc, aefd->"(x,gate,conj(x))
    return e[1]
end

function energy3(x,gate)
    return ncon([x, gate, conj(x)], [[1 2 3 4], [5 6 2 3], [1 5 6 4]])
end

function energy4(x,gate)
    return ncon((x, gate, conj(x)), ([1 2 3 4], [5 6 2 3], [1 5 6 4]))
end

function norm(x)
    return @tensor x[1 2 3 4]*conj(x[1 2 3 4])
end

function norm2(x)
    y = reshape(x, prod(size(x)))
    norm = ein"i,i -> "(y,y)[]
    return norm
end

function norm3(x)
    return ncon([x, conj(x)], [[1 2 3 4], [1 2 3 4]])
end

function norm4(x)
    return ncon((x, conj(x)), ([1 2 3 4], [1 2 3 4]))
end

function iDMRG(Ac)
    gate = isingTBG(1, 1)
    return energy2(Ac, gate)
end

Ac = randn(1,2,2,1)
gate = isingTBG(1., 0.)
# energy(Ac,gate) = @tensor Ac[1 2 3 4]*gate[5 6 2 3]*conj(Ac)[1 5 6 4]

# energy(Ac) = Ac
println(energy4(Ac, gate))
println("EYY")
# gradient(x->tr(x*x'), rand(2,2))
gradient(x -> energy2(x, gate), Ac)
gradient(x -> energy4(x, gate), Ac)

res = Nothing
let energy = x -> real(energy4(x, gate) / norm3(x))
    global res = optimize(energy,
        Δ -> Zygote.gradient(energy, Δ)[1], Ac, LBFGS(m = 20), inplace = false)
end