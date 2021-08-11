using TensorKit
# using Zygote
# using Flux
# using Zygote: forwarddiff
using Optim
# using ChainRulesCore
using FiniteDifferences

Lx = Ly = 2
chiB = 2
d = 2

pepsTensors = rand(Float64, Lx, Ly, chiB, chiB, d, chiB, chiB);

function cnorm(pepsTensors)
    tensor = TensorMap(pepsTensors[1,1,:,:,:,:,:], ℝ^chiB ⊗ ℝ^chiB ⊗ ℝ^d, ℝ^chiB ⊗ ℝ^chiB)
    norm = @tensor tensor[1,2,3,4,5]*conj(tensor[1,2,3,4,5])

    # tensor = pepsTensors[1,1,:,:,:,:,:]
    # tensor = reshape(tensor, prod(size(tensor)))
    # norm = tensor'*tensor
    return norm
end

optimmethod = LBFGS(m = 20);
optimargs = (Optim.Options(f_tol = 1e-3, show_trace = true), );
res = nothing
let energy = x -> real(cnorm(x))
    global res = optimize(energy, Δ -> grad(central_fdm(5, 1), cnorm, Δ)[1], pepsTensors, optimmethod, inplace = false, optimargs...);
end