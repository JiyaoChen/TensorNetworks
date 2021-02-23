mutable struct Model{A<:AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}, S<:EuclideanSpace}
    H :: Vector{A}  # vector of (site-dependent) hamiltonian mpo's
    Q :: Vector{S}  # vector of link quantum numbers (for initial state and iDMRG)
    P :: Dict{Any, Any}  # dictionary for list of parameters used in the algorithm
end

# convenient wrapper to pass function returns to the default constructor
function Model(genH::Function, genQ::Function, genP::Function)
    Model(genH(genP()), genQ(genP()), genP())
end