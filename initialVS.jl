function generateInitialVS(P::Dict)
    vectorspaces = [P["sym"](i => 1) for i = 0 : P["length"]]
    return vectorspaces
end