
using IterTools: iterated

# function CTMRGStep(C1, T1, C2, T2, C3, T3, C4, T4, (iPEPS, chiE, truncBelowE, d))

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
end

# @Zygote.nograd StopFunction
function (st::StopFunction)(state)
    # @info state
    st.counter += 1
    st.counter > st.maxit && return true

    chiE = size(state[9], 4);
    C1, C2, C3, C4 = state[[1, 3, 5, 7]];
    
    newSingularValues = zeros(Float64, C1.Lx, C1.Ly, 4, chiE);
    foreach(keys(C1.tensorDict)) do tensorKey
        Λ1 = svd(C1.tensorDict[tensorKey]).S;
        Λ2 = svd(C2.tensorDict[tensorKey]).S;
        Λ3 = svd(C3.tensorDict[tensorKey]).S;
        Λ4 = svd(C4.tensorDict[tensorKey]).S;
        newSingularValues[tensorKey[1], tensorKey[2], 1, 1 : length(Λ1)] = Λ1;
        newSingularValues[tensorKey[1], tensorKey[2], 2, 1 : length(Λ2)] = Λ2;
        newSingularValues[tensorKey[1], tensorKey[2], 3, 1 : length(Λ3)] = Λ3;
        newSingularValues[tensorKey[1], tensorKey[2], 4, 1 : length(Λ4)] = Λ4;
        # @info Λ1, Λ2, Λ3, Λ4
    end

    # vals = state[3]
    diff = norm(newSingularValues - st.oldvals)
    println(diff)
    diff <= st.tol && return true
    st.oldvals = newSingularValues

    return false
end

# function f(x)

#     y = x[3];
#     return (x[1], x[2], y^2 - 3*y + 4)

# end

# stopFun = StopFunction(1., 0, 1e-3, 1000)
# guess = [0.0, 0.0, 1.2];
# for state in iterated(x -> f(x), guess)
#     stopFun(state) && return state
# end