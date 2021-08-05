
using IterTools: iterated

function fixedPoint(f, guess, init, stopFunc)
    for state in iterated(x -> f(x, init), guess)
        stopFunc(state) && return state
    end
end

mutable struct StopFunction{T,S}
    oldvals::T
    counter::Int
    tol::S
    maxit::Int
    chiE::Int
end

function getSingularValues(cornerTensors, idx, idy, chiE)
    singularValues = zeros(Float64, chiE, 0);
    foreach(cornerTensors) do C
        singVals = svd(C[idx, idy]).S;
        singularValues = hcat(singularValues, vcat(singVals, zeros(chiE - length(singVals))));
    end
    return singularValues;
end

@Zygote.nograd StopFunction
function (stopFunc::StopFunction)(stateCTMRG)
    
    stopFunc.counter += 1
    stopFunc.counter > stopFunc.maxit && return true

    chiE = stopFunc.chiE;
    C1, C2, C3, C4 = stateCTMRG[[1, 3, 5, 7]];
    
    # cornerTensors = (stateCTMRG[[1, 3, 5, 7]]);
    # newSingularValues = [getSingularValues(cornerTensors, idx, idy, chiE) for idx = 1 : Lx, idy = 1 : Ly];
    # println(newSingularValues)
    # println(size(newSingularValues))

    newSingularValues = zeros(Float64, C1.Lx, C1.Ly, 4, chiE);
    for idx = 1 : Lx, idy = 1 : Ly
        Λ1 = svd(C1.tensorArray[idx, idy]).S;
        Λ2 = svd(C2.tensorArray[idx, idy]).S;
        Λ3 = svd(C3.tensorArray[idx, idy]).S;
        Λ4 = svd(C4.tensorArray[idx, idy]).S;
        newSingularValues[idx, idy, 1, 1 : length(Λ1)] = Λ1;
        newSingularValues[idx, idy, 2, 1 : length(Λ2)] = Λ2;
        newSingularValues[idx, idy, 3, 1 : length(Λ3)] = Λ3;
        newSingularValues[idx, idy, 4, 1 : length(Λ4)] = Λ4;
        # @info Λ1, Λ2, Λ3, Λ4
    end

    diff = norm(newSingularValues - stopFunc.oldvals)
    @printf("convergence CTMRG step %d : %0.6e\n", stopFunc.counter, diff)
    diff <= stopFunc.tol && return true
    stopFunc.oldvals = newSingularValues

    return false
end

#=
    define custom adjoint for reverse-mode AD through fixed point CTMRG routine
=#

# fixedpointbackward(next, (c,t,vals), (a, χ, d))

function fixedPointBackward(next, CTMRGTensors, (iPEPS, chiE, truncBelowE))
    
    _, back = Zygote.pullback(next, CTMRGTensors, (iPEPS, chiE, truncBelowE));
    back1 = x -> back(x)[1];
    back2 = x -> back(x)[2];

    function backΔ(Δ)
        grad = back2(Δ)[1]
        for g in take(imap(back2,drop(iterated(back1, Δ),1)),100)
            grad .+= g[1]
            ng = norm(g[1])
            if ng < 1e-7
                break
            elseif ng > 10
                println("backprop not converging")
                # try to minimise damage by scaling to small
                grad ./= norm(grad)
                grad .*= 1e-4
                break
            end
        end
        (grad, nothing, nothing)
    end
    return backΔ
end

fixedPointAD(f, guess, init, stopFunc) = fixedPoint(f, guess, init, stopFunc);

@Zygote.adjoint function fixedPointAD(f, guess, init, stopFunc)
    r = fixedPoint(f, guess, init, stopFunc);
    return r, Δ -> (nothing, nothing, fixedPointBackward(f, r, n)(Δ), nothing);
end