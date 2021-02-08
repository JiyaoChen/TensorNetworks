
using LinearAlgebra
using KrylovKit
using TensorKit
using TensorOperations

vP = ℤ₂Space(0 => 1, 1 => 1);
vV = ℤ₂Space(0 => 1);

vL = ℤ₂Space(0 => 2, 1 => 1);
vR = ℤ₂Space(0 => 2, 1 => 1);

# Pauli opertators
Id = Matrix{ComplexF64}(I,2,2);
X = 1/2 * [ 0 +1 ; +1 0 ];
Y = 1/2 * [ 0 -1im ; +1im 0 ];
Z = 1/2 * [ +1 0 ; 0 -1 ];

# parameters Ising model
J = 1.0;
h = 0.9;
maxNumSteps = convert(Int64,1e1);
χ = 100;

# generate the Heisenberg MPO
ham_arr = zeros(Complex{Float64}, dim(vP), dim(vL), dim(vR), dim(vP));
ham_arr[:,1,1,:] = Id;
ham_arr[:,2,2,:] = Id;
ham_arr[:,1,3,:] = -J*X;
ham_arr[:,3,2,:] = X;
ham_arr[:,1,2,:] = -h*Z;

mpo = TensorMap(ham_arr, vP*vL, vR*vP)

gammaList = fill(TensorMap(ones, vP*vV, vV), 2)
lambdaList = fill(TensorMap(ones, vV, vV), 2)

EL = TensorMap(ones, vV, vL*vV)
ER = TensorMap(ones, vR*vV, vV)

groundStateEnergy = zeros(Float64, maxNumSteps, 5);

for i = 1 : maxNumSteps

    groundStateEnergy[i,1] = i;

    for i_bond = 1 : 2

        # choose unit cell arrangement
        ia = mod(i_bond + 0,2) + 1;
        ib = mod(i_bond + 1,2) + 1;
        ic = mod(i_bond + 2,2) + 1;

        # # get dimensions of tensors to optimize
        # chia = size(gammaList[ib],2);
        # chic = size(gammaList[ic],3);

        # construct initial wave function
        @tensor theta[:] := lambdaList[ia][-3 1] * gammaList[ib][-2 1 2] * gammaList[ic][-1 2 -4];
        # thetaInit = reshape(thetaInit,(chia*d*d*chic));
        # thetaInit = thetaInit ./ sqrt(abs(dot(thetaInit,thetaInit)));

        # # store previous eivenvalue
        # prevEigenVal = eigenVal;
        # prevEigenVec = thetaInit;

        # function to apply the Hamiltonian to the wave function
        function applyH(X, EL, mpo, ER)
            @tensor Y[:] := EL[-3 3 1] * X[4 2 1 6] * mpo[-2 3 5 2] * mpo[-1 5 7 4] * ER[7 6 -4];
            return Y
        end
        
        # optimize wave function
        eigenVal, eigenVec = eigsolve(x -> applyH(x, EL, mpo, ER), theta, 1, :SR, Lanczos());
        thetaP = eigenVec[1];
        
        #  perform SVD and truncation to desired bond dimension
        U, S, V, ϵ = tsvd(thetaP, (2,3), (1,4), trunc = truncdim(χ));
        V = permute(V, (2, 1, 3));
        print(eigenVal,"\n")
        print(ϵ,"\n")
        print(U)
        # print(S)
        # print(V)

        # update environment
        @tensor EL[:] = EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
        @tensor ER[:] = ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);
        # @tensor newEL[:] := EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
        # @tensor newER[:] := ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);
        # global EL = newEL;
        # global ER = newER;

        # obtain the new tensors for MPS
        # lambdaList[ib] = S ./ sqrt(sum(diag(S.^2)));
        @tensor gammaList[ib][:] = inv(lambdaList[ia][-2 1]) * U[-1 1 2] * lambdaList[ib][2 -3];
        gammaList[ic] = V;

    end

end