mutable struct MPO{A<:AbstractTensorMap{S,2,2} where S<:EuclideanSpace}
    mpo :: Vector{Union{Missing,A}}
end