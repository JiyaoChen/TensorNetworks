using TensorKit

function generateParameters()
    parameters = Dict{Any,Any}()
    push!(parameters, "χ" => 100)
    push!(parameters, "length" => 20)
    push!(parameters, "tol" => 1e-12)
    push!(parameters, "sym" => U1Space)
    push!(parameters, "spin" => 1/2)
    return parameters


    # modelParams = [("χ", 100),("length", 10)]
    # A = Dict{Any,Any}(modelParams)
end 