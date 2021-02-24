using TensorKit
using KrylovKit

function generateParameters()

    parameters = Dict{Any,Any}()
    push!(parameters, "χ" => 256)
    push!(parameters, "sweeps" => 8)
    push!(parameters, "length" => 100)
    
    # settings for the eigensolver
    push!(parameters, "solver" => Arnoldi)
    push!(parameters, "tol" => 1e-4)
    push!(parameters, "maxiter" => 4)
    push!(parameters, "krylovdim" => 3)
    
    # TODO -- move to Hamiltonian?
    push!(parameters, "sym" => U1Space)
    push!(parameters, "spin" => 1/2)
    push!(parameters, "eltype" => Float64)
    
    return parameters

end 