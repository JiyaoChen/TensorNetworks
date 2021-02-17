# construct initial wave function
function initialWF(L, G1, G2)
    @tensor Y[-1 -2 -3; -4] := L[-3 1] * G1[-2 1 2] * G2[-1 2 -4];
    return Y
end
# 
function new_gamma(L1, U, L2)
    @tensor Y[-1 -2; -3] := pinv(L1)[-2 1] * U[-1 1 2] * L2[2 -3];
    return Y
end
# 
function update_EL(EL, U, mpo)
    @tensor Y[-1; -2 -3] := EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
    return Y
end
#
function update_ER(ER, V, mpo)
    @tensor Y[-1 -2; -3] := ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);
    return Y
end
# function to apply the Hamiltonian to the wave function
function applyH(X, EL, mpo, ER)
    @tensor Y[-1 -2 -3; -4] := EL[-3 3 1] * X[4 2 1 6] * mpo[-2 3 5 2] * mpo[-1 5 7 4] * ER[7 6 -4];
    return Y
end