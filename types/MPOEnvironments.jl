mutable struct MPOEnvironments{A<:AbstractTensorMap{S,1,2} where S<:EuclideanSpace, B<:AbstractTensorMap{S,2,1} where S<:EuclideanSpace}
    mpoEnvL :: Vector{Union{Missing,A}}
    mpoEnvR :: Vector{Union{Missing,B}}
end

function MPOEnvironments(mps::MPS, mpo::MPO)
end