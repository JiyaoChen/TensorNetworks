include("../engines/DMRG_contractions.jl")

mutable struct MPOEnvironments{A<:MPOEnvLTensor, B<:MPOEnvRTensor}
    mpoEnvL :: Vector{A}
    mpoEnvR :: Vector{B}
end

function MPOEnvironments(mps::MPS, mpo::MPO)
    mpoEnvL = Vector{MPOEnvLTensor}(undef, length(mps.ACs))
    mpoEnvR = Vector{MPOEnvRTensor}(undef, length(mps.ACs))

    # weight all available sectors with 1.0
    mpoEnvL[1] = TensorMap(ones, eltype(mpo.mpo[1]), space(mps.ACs[1],1), space(mpo.mpo[1], 1) ⊗ space(mps.ACs[1],1))
    mpoEnvR[end] = TensorMap(ones, eltype(mpo.mpo[1]), space(mps.ARs[end],3)' ⊗ space(mpo.mpo[end], 3)', space(mps.ARs[end],3)')
    for i = length(mpoEnvR) : -1 : 2
        @tensor mpoEnvR[i-1][-1 -2; -3] := mps.ARs[i][-1 3 1] * mpoEnvR[i][1 2 5] * mpo.mpo[i][-2 4 2 3] * conj(mps.ARs[i][-3 4 5])
    end

    MPOEnvironments(mpoEnvL, mpoEnvR)
end

# struct and constructors for infinite MPS
mutable struct InfiniteMPOEnvironments{A<:MPOEnvLTensor, B<:MPOEnvRTensor}
    mpoEnvL :: PeriodicArray{A,1}
    mpoEnvR :: PeriodicArray{B,1}
end

function InfiniteMPOEnvironments(mps::InfiniteMPS, mpo::MPO)
    mpoEnvL = PeriodicArray{MPOEnvLTensor,1}(undef, length(mps.ACs))
    mpoEnvR = PeriodicArray{MPOEnvRTensor,1}(undef, length(mps.ACs))

    # compute dominant left environment eigenvector
    theta = TensorMap(randn, space(mps.ACs[1],1), space(mpo.mpo[1],1) ⊗ space(mps.ACs[1],1))
    theta = permute(theta, (1,2,3), ())
    eigenVal, eigenVec = 
        eigsolve(theta,1, :LM, Lanczos(tol=1e-6)) do x
            contractTWL(x, mps, mpo.mpo)
        end
    print("left eigenvalue:", eigenVal, "\n")
    mpoEnvL[1] = permute(eigenVec[1], (1,), (2,3))

    theta = TensorMap(randn, space(mps.ACs[end],1) ⊗ space(mpo.mpo[end],1),  space(mps.ACs[end],1))
    theta = permute(theta, (1,2,3), ())
    eigenVal, eigenVec = 
        eigsolve(theta,1, :LM, Lanczos(tol=1e-6)) do x
            contractTWR(x, mps, mpo.mpo)
        end
    print("right eigenvalue:", eigenVal, "\n")
    mpoEnvR[end] = permute(eigenVec[1], (1,2), (3,))
    # for i = length(mpoEnvR) : -1 : 2
    #     @tensor mpoEnvR[i-1][-1 -2; -3] := mps.ARs[i][-1 3 1] * mpoEnvR[i][1 2 5] * mpo.mpo[i][-2 4 2 3] * conj(mps.ARs[i][-3 4 5])
    # end

    InfiniteMPOEnvironments(mpoEnvL, mpoEnvR)
end