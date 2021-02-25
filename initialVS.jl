function generateInitialVS(P::Dict)
    
    # vectorspaces = [P["sym"](i => 1) for i = 0 : P["length"]]
    # vectorspaces = [P["sym"](0 => 1) for i = 0 : P["length"]]

    qns = Array{Int64}(undef,P["length"]+1)
    ctr = 0
    for i = 1 : length(qns)
        qns[i] = ctr
        ctr += mod(i,2)*floor((2*P["spin"]+1)/2)
    end
    vectorspaces = [P["sym"](qns[i] => 1) for i = 1 : length(qns)]
    
    return vectorspaces

end

function generateInitialVSNoSym(P::Dict)

    vectorspaces = [ℂ^1 for i = 1 : P["length"]+1]
    
    return vectorspaces

end

function generateInitialVSSU2(P::Dict)

    vectorspaces = [SU2Space(mod(i,2)*P["spin"]=>1) for i = 0 : P["length"]]
    
    return vectorspaces

end