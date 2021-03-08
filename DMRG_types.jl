module DMRG_types

    using TensorKit

    # MPS tensors
    const GenericMPSTensor{S,N} = AbstractTensorMap{S,N,1} where {S<:EuclideanSpace,N}
    const MPSTensor{S} = GenericMPSTensor{S,2} where {S<:EuclideanSpace}
    const MPSBondTensor{S} = AbstractTensorMap{S,1,1} where {S<:EuclideanSpace}

    # MPO tensors
    const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}
    const MPOEnvLTensor{S} = AbstractTensorMap{S,1,2} where {S<:EuclideanSpace}
    const MPOEnvRTensor{S} = AbstractTensorMap{S,2,1} where {S<:EuclideanSpace}

    include("types/MPO.jl")
    include("types/Model.jl")
    include("types/PeriodicArray.jl")
    include("types/MPS.jl")
    include("types/MPOEnvironments.jl")

end