function newGuess(Spr, Vdag, S, U)
    @tensor Y[-1 -2 -3; -4] := Spr[-1 1] * Vdag[1 -2 2] * pinv(S)[2 3] * U[3 -3 4] * Spr[4 -4]
    return Y
end

function newGuess(shiftIrrep, Spr, Vdag, S, U)
    @tensor rPart[-1 -2; -3] := pinv(S)[-1 1] * U[1 -2 2] * Spr[2 -3]
    rPart = shiftVirtSpaceMPS(rPart, shiftIrrep)
    @tensor Y[-1 -2 -3; -4] := Spr[-1 1] * Vdag[1 -2 2] * rPart[2 -3 -4];
    # @tensor Y[-1 -2 -3; -4] := Spr[-1 1] * Vdag[1 -2 2] * pinv(S)[2 3] * U[3 -2 4] * Spr[4 -3]
    return Y
end

function newGuess(shiftIrrep, maxIrrepL, maxIrrepR, Spr, Vdag, S, U)
    @tensor rPart[-1 -2; -3] := pinv(S)[-1 1] * U[1 -2 2] * Spr[2 -3]
    rPart = shiftVirtSpaceMPS(rPart, shiftIrrep, maxIrrepL, maxIrrepR)
    @tensor Y[-1 -2 -3; -4] := Spr[-1 1] * Vdag[1 -2 2] * rPart[2 -3 -4];
    # @tensor Y[-1 -2 -3; -4] := Spr[-1 1] * Vdag[1 -2 2] * pinv(S)[2 3] * U[3 -2 4] * Spr[4 -3]
    return Y
end
 
function update_EL(EL, U, mpo)
    @tensor Y[-1; -2 -3] := EL[5 3 1] * U[1 2 -3] * mpo[3 4 -2 2] * conj(U[5 4 -1])
    return Y
end

function update_ER(ER, Vdag, mpo)
    @tensor Y[-1 -2; -3] := Vdag[-1 2 1] * ER[1 3 5] * mpo[-2 4 3 2] * conj(Vdag[-3 4 5])
    return Y
end

function applyH(X, EL, mpo1, mpo2, ER)
    Y = zero(X)
    @tensor Y[-1 -2 -3 -4] := EL[-1 2 1] * X[1 3 4 6] * mpo1[2 -2 5 3] * mpo2[5 -3 7 4] * ER[6 7 -4]
    return Y
end

function applyH0(X, EL, ER)
    @tensor X[-1 -2] := EL[-1 3 1] * X[1 2] * ER[2 3 -2]
    return X
end

function applyH1(X, EL, mpo, ER)
    @tensor X[-1 -2 -3] := EL[-1 2 1] * X[1 3 4] * mpo[2 -2 5 3] * ER[4 5 -3]
    return X
end

function applyH2(X, EL, mpo1, mpo2, ER)
    @tensor X[-1 -2 -3 -4] := EL[-1 2 1] * X[1 3 5 6] * mpo1[2 -2 4 3] * mpo2[4 -3 7 5] * ER[6 7 -4]
    return X
end

function contractTWL(X, mps, mpo)
    Y = X
    for i = 1 : length(mps.ACs)
        # println(i)
        @tensor Y[-1 -2 -3] := Y[5 3 1] * mps.ACs[i][1 2 -3] * mpo[i][3 4 -2 2] * conj(mps.ACs[i][5 4 -1])
    end
    println(Y)
    shiftTensor = TensorKit.isomorphism(space(Y,3)' ⊗ space(Y,3), space(X,1))
    @tensor X[-1 -2 -3] := Y[1 -2 2] * shiftTensor[2 3 -3] * conj(shiftTensor[1 3 -1])

    println(X)
    return X
end

function contractTWR(X, mps, mpo)
    Y = X
    for i = length(mps.ACs) : -1 : 1
        # println(i)
        @tensor Y[-1 -2 -3] := mps.ACs[i][-1 2 1] * mpo[i][-2 4 3 2] * conj(mps.ACs[i][-3 4 5]) * Y[1 3 5]
    end
    println(Y)
    shiftTensor = TensorKit.isomorphism(space(X,3)' ⊗ space(X,3), space(Y,1))
    @tensor X[-1 -2 -3] := Y[2 -2 1] * shiftTensor[-1 3 2] * conj(shiftTensor[-3 3 1])

    println(X)
    return X
end