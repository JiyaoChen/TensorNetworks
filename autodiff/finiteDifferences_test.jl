using TensorKit
# using Zygote
# using Flux
# using Zygote: forwarddiff
using Optim
# using ChainRulesCore
using FiniteDifferences

chiB = 6
d = 2

entries = rand(Float64, chiB, chiB, d, chiB, chiB);
entries /= norm(entries)

function cnorm(entries)
    
    tensor = TensorMap(entries, ℝ^chiB ⊗ ℝ^chiB ⊗ ℝ^d, ℝ^chiB ⊗ ℝ^chiB)
    norm = @tensor tensor[1,2,3,4,5]*conj(tensor[1,2,3,4,5])

    return (norm-3)^2+1  # optimal tensor has norm == 3
end

optimmethod = LBFGS(m = 20);
optimargs = (Optim.Options(f_tol = 1e-16, show_trace = true), );
res = nothing
let energy = x -> real(cnorm(x))
    global res = optimize(energy, Δ -> grad(central_fdm(5, 1), cnorm, Δ)[1], entries, optimmethod, inplace = false, optimargs...);
end
minimized_norm = reshape(res.minimizer,chiB^4*d)'*reshape(res.minimizer,chiB^4*d)
@show minimized_norm
0;  # suppres standard REPL out