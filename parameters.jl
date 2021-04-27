using TensorKit
using KrylovKit

function generateParameters()

    # this might not be the best choice for the dict entries
    parameters = Dict{Any,Any}()

    # settings related to the sweeping procedure
    # χ = [fill(χ,4) for χ in [2 4 8 16 32 64 128 256 512 1024 1024 1024]]
    χ = [fill(χ,2) for χ in [10]]
    push!(parameters, "χ" => [(χ...)...])
    push!(parameters, "length" => 3)

    push!(parameters, "Lx" => 3)
    push!(parameters, "Ly" => 3)
    
    # settings for the eigensolver
    push!(parameters, "solver" => Lanczos)
    push!(parameters, "tol" => 1e-1)
    push!(parameters, "maxiter" => 1)
    push!(parameters, "krylovdim" => 4)
    
    # TODO -- move this block to Hamiltonian?
    push!(parameters, "sym" => U1Space)
    push!(parameters, "spin" => 0.5)
    push!(parameters, "eltype" => Complex{Float64})
    
    return parameters

end 