
using LinearAlgebra
using KrylovKit
using TensorKit
using TensorOperations

Base.run(`clear`)
for var in names(Main)
    try
        eval(parse("$var=missing"))
    catch e
    end
end
GC.gc()

vP = ℤ₂Space(0 => 1, 1 => 1);
vV = ℤ₂Space(0 => 1);
vL = ℤ₂Space(0 => 2, 1 => 1);
vR = ℤ₂Space(0 => 2, 1 => 1);

# vP = ℂ^2;
# vV = ℂ^1;
# vL = ℂ^3;
# vR = ℂ^3;

# Pauli opertators
Id = Matrix{ComplexF64}(I,2,2);
X = 1/2 * [ 0 +1 ; +1 0 ];
Y = 1/2 * [ 0 -1im ; +1im 0 ];
Z = 1/2 * [ +1 0 ; 0 -1 ];

# parameters Ising model
J = 1.0;
h = 0.9;
maxNumSteps = convert(Int64,1e1);
χ = 4;

# generate the Heisenberg MPO
ham_arr = zeros(Complex{Float64}, dim(vP), dim(vL), dim(vR), dim(vP));
ham_arr[:,1,1,:] = Id;
ham_arr[:,2,2,:] = Id;
ham_arr[:,1,3,:] = -J*X;
ham_arr[:,3,2,:] = X;
ham_arr[:,1,2,:] = -h*Z;

mpo = TensorMap(ham_arr, vP*vL, vR*vP)

# gammaList = fill(TensorMap(ones, vP*vV, vV), 2)
# lambdaList = fill(TensorMap(ones, vV, vV), 2)
# EL = TensorMap(ones, vV, vL*vV)
# ER = TensorMap(ones, vR*vV, vV)

# initialize MPS
gammaTensor_arr = zeros(ComplexF64, dim(vP), dim(vV), dim(vV));
gammaTensor_arr[1,1,1] = 1;
gammaTensor = TensorMap(gammaTensor_arr, vP*vV, vV);
gammaList = fill(gammaTensor,2);

lambdaTensor_arr = zeros(ComplexF64, dim(vV), dim(vV));
lambdaTensor_arr[1,1] = 1;
lambdaTensor = TensorMap(lambdaTensor_arr, vV, vV);
lambdaList = fill(lambdaTensor,2);

# initiliaze EL and ER
EL_arr = zeros(ComplexF64,dim(vV),dim(vL),dim(vV));
EL_arr[1,1,1] = 1;
EL = TensorMap(EL_arr, vV, vL*vV);

ER_arr = zeros(ComplexF64,dim(vR),dim(vV),dim(vV));
ER_arr[dim(vR),1,1] = 1;  # for ℂ
ER_arr[2,1,1] = 1;  # for ℤ₂
ER = TensorMap(ER_arr, vR*vV, vV);

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
        # println("LA")
        # println((lambdaList[ia]))
        # println("GB")
        # println((gammaList[ib]))
        # println("GC")
        # println((gammaList[ic]))
        @tensor theta[-1 -2 -3; -4] := lambdaList[ia][-3 1] * gammaList[ib][-2 1 2] * gammaList[ic][-1 2 -4];
        # thetaInit = reshape(thetaInit,(chia*d*d*chic));
        # thetaInit = thetaInit ./ sqrt(abs(dot(thetaInit,thetaInit)));

        # # store previous eivenvalue
        # prevEigenVal = eigenVal;
        # prevEigenVec = thetaInit;

        # function to apply the Hamiltonian to the wave function
        function applyH(X, EL, mpo, ER)
            # println("EL -- ", EL)
            # println("Theta -- ", X)
            @tensor Y[-1 -2 -3; -4] := EL[-3 3 1] * X[4 2 1 6] * mpo[-2 3 5 2] * mpo[-1 5 7 4] * ER[7 6 -4];
            return Y
        end
        # function applyH(X, EL, mpo, ER)
        #     @tensor Y[:] := EL[-3 3 1] * X[4 2 1 6] * mpo[-2 3 5 2] * mpo[-1 5 7 4] * ER[7 6 -4];
        #     return Y
        # end

        # applyH(theta)
        
        # optimize wave function
        eigenVal, eigenVec = let EL = EL, mpo = mpo, ER = ER
            eigsolve(theta,1, :SR, Lanczos()) do x
                applyH(x, EL, mpo, ER)
            end
        end
        # eigenVal, eigenVec = eigsolve(x -> applyH(x), theta, 1, :SR, Lanczos());
        # eigenVal, eigenVec = eigsolve(x -> applyH(x, EL, mpo, ER), theta, 1, :SR, Lanczos());
        thetaP = eigenVec[1];
        
        #  perform SVD and truncation to desired bond dimension
        U, S, V, ϵ = tsvd(thetaP, (2,3), (1,4), trunc = truncdim(χ));
        # print("V old ---- ", V)
        V = permute(V, (2, 1), (3, ));
        print(eigenVal[1],"\n")
        print(ϵ,"\n")
        # print(U)
        print(S)
        # print("V new ---- ", V)

        # update environment
        # println("EL")
        # println((EL))
        # println("U")
        # println((U))
        # println("MPO")
        # println((mpo))
        # @tensor EL[:] = EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
        # @tensor ER[:] = ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);

        # @tensor newEL[:] := EL[-1 -2 1] * U[-4 1 -3];
        # @tensor newEL[:] := U[2 -1 -3] * mpo[-4 -5 -2 2];
        # @tensor newEL[:] := mpo[4 -3 -2 -4] * conj(U[4 -5 -1]);
        # @tensor newEL[:] := EL[-5 3 -1] * mpo[-4 3 -2 -3];
        # @tensor newEL[:] := EL[5 -1 -2] * conj(U[-3 5 -4]);

        @tensor newEL[-1; -2 -3] := EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
        @tensor newER[-1 -2; -3] := ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);
        # @tensor newEL[:] := EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
        # @tensor newER[:] := ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);
        global EL = newEL;
        global ER = newER;

        # obtain the new tensors for MPS
        lambdaList[ib] = S;
        @tensor newgamma[-1 -2; -3] := (lambdaList[ia][-2 1]) * U[-1 1 2] * lambdaList[ib][2 -3];

        # println("LB --- ", lambdaList[ib])
        # println("U --- ", U)
        # println("NG ", newgamma)

        # println("new Gamma -- ", newgamma)
        # println("old Gamma -- ", gammaList[ib])
        global gammaList[ib] = newgamma;
        global gammaList[ic] = V;

    end

end