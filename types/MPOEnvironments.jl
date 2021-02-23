mutable struct MPOEnvironments{A<:MPOEnvLTensor, B<:MPOEnvRTensor}
    mpoEnvL :: Vector{A}
    mpoEnvR :: Vector{B}
end

function MPOEnvironments(mps::MPS, mpo::MPO)
    mpoEnvL = Vector{MPOEnvLTensor}(undef, length(mps.ACs))
    mpoEnvR = Vector{MPOEnvRTensor}(undef, length(mps.ACs))

    # weight all available sectors with 1.0
    mpoEnvL[1] = TensorMap(ones, space(mps.ACs[1],1), space(mpo.mpo[1], 1) ⊗ space(mps.ACs[1],1))
    mpoEnvR[end] = TensorMap(ones, space(mps.ARs[end],3)' ⊗ space(mpo.mpo[end], 3)', space(mps.ARs[end],3)')
    for i = length(mpoEnvR) : -1 : 2
        @tensor mpoEnvR[i-1][-1 -2; -3] := mps.ARs[i][-1 3 1] * mpoEnvR[i][1 2 5] * mpo.mpo[i][-2 4 2 3] * conj(mps.ARs[i][-3 4 5])
    end

    MPOEnvironments(mpoEnvL, mpoEnvR)
end