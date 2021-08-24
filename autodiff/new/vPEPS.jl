# # the basic struct used to define a tensorial peps object on a given unit cell
# mutable struct pepsUnitCell{T}
#     Lx::Int64
#     Ly::Int64
#     tensorArray::Array{T, 2}
#     unitCellLayout::Matrix{Int64}
# end

module vPEPS

    # load packages
    using Base.Iterators: drop, take
    using IterTools: imap, iterated
    using LinearAlgebra
    using OMEinsum
    using Optim
    using Printf
    using Zygote

    # make public
    export optimizePEPS
    export computeEnergy

    include("CTMRG.jl")
    include("customAdjoints.jl")
    include("expectationValues.jl")
    
    # include optimization methods
    include("vPEPS_methods.jl")

end