
function fixedPoint(f, guess, init, stopFunc)
    for state in iterated(x -> f(x, init), guess)
        stopFunc(state) && return state
    end
end

mutable struct StopFunction{T,S}
    Lx::Int
    Ly::Int
    oldvals::T
    counter::Int
    tol::S
    maxit::Int
    chiE::Int
end

# @Zygote.nograd getSingularValues
function getSingularValues(C1, C2, C3, C4, idx, idy, chiE)
    _, Λ1, _ = svd(C1[idx, idy]);
    _, Λ2, _ = svd(C2[idx, idy]);
    _, Λ3, _ = svd(C3[idx, idy]);
    _, Λ4, _ = svd(C4[idx, idy]);
    return vcat(Λ1, zeros(chiE - length(Λ1)), Λ2, zeros(chiE - length(Λ2)), Λ3, zeros(chiE - length(Λ3)), Λ4, zeros(chiE - length(Λ4)));
end

@Zygote.nograd StopFunction
function (stopFunc::StopFunction)(stateCTMRG)
    
    stopFunc.counter += 1
    stopFunc.counter > stopFunc.maxit && return true

    Lx = stopFunc.Lx;
    Ly = stopFunc.Ly;
    chiE = stopFunc.chiE;
    C1, C2, C3, C4 = stateCTMRG[[1, 3, 5, 7]];
    newSingularValues = [getSingularValues(C1, C2, C3, C4, idx, idy, chiE) for idx = 1 : Lx, idy = 1 : Ly];

    diff = norm(newSingularValues .- stopFunc.oldvals);
    # @printf("convergence CTMRG step %d : %0.6e\n", stopFunc.counter, diff)
    # println("convergence CTMRG: ", diff)
    diff <= stopFunc.tol && return true;
    stopFunc.oldvals = newSingularValues;

    return false
end

#=
    define custom adjoint for reverse-mode AD through fixed point CTMRG routine
=#

function fixedPointBackward(f, CTMRGTensors, (iPEPS, unitCellLayout, chiE, truncBelowE))
    
    _, back = Zygote.pullback(f, CTMRGTensors, (iPEPS, unitCellLayout, chiE, truncBelowE));
    back1 = x -> back(x)[1];
    back2 = x -> back(x)[2];

    function backΔ(Δ)
        grad = back2(Δ)[1];
        for g in take(imap(back2, drop(iterated(back1, Δ), 1)), 100)
            # println(g)
            grad .+= g[1]
            ng = norm(g[1])
            # println(ng)
            if ng < 1e-7
                println("backprop converged")
                break
            elseif ng > 10
                println("backprop not converging")
                # try to minimise damage by scaling to small
                grad ./= norm(grad)
                grad .*= 1e-4
                break
            elseif isnan(ng)
                println("backprop NaN")
                break
            end
        end
        return (grad, nothing, nothing, nothing)
    end
    return backΔ

end

fixedPointAD(f, guess, init, stopFunc) = fixedPoint(f, guess, init, stopFunc);

@Zygote.adjoint function fixedPointAD(f, guess, init, stopFunc)
    r = fixedPoint(f, guess, init, stopFunc);
    return r, Δ -> (nothing, nothing, fixedPointBackward(f, r, init)(Δ), nothing);
end