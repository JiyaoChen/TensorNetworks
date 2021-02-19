function newGuess(Spr, Vdag, S, U)
    @tensor Y[-1 -2 -3; -4] := Spr[-1 1] * Vdag[1 -2 2] * pinv(S)[2 3] * U[3 -3 4] * Spr[4 -4]
    return Y
end
 
function update_EL(EL, U, mpo)
    @tensor Y[-1 -2; -3] := EL[5 3 1] * U[1 2 -3] * mpo[3 4 -2 2] * conj(U[5 4 -1])
    return Y
end

function update_ER(ER, Vdag, mpo)
    @tensor Y[-1; -2 -3] := Vdag[-1 2 1] * ER[1 3 5] * mpo[-2 4 3 2] * conj(Vdag[-3 4 5])
    return Y
end

function applyH(X, EL, mpo, ER)
    @tensor Y[-1 -2 -3; -4] := EL[-1 2 1] * X[1 3 4 6] * mpo[2 -2 5 3] * mpo[5 -3 7 4] * ER[6 7 -4]
    return Y
end