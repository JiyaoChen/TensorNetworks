
using LinearAlgebra
using KrylovKit
using Printf
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

# set symmetry
setSym = "Z2";

# parameters Ising model
J = 1.0;
h = 1.0;
maxNumSteps = convert(Int64,1e2);
χ = 10;
if setSym == "NO"
    vP = ℂ^2;
    vV = ℂ^1;
    vL = ℂ^3;
    vR = ℂ^3;
elseif setSym == "Z2"
    vP = ℤ₂Space(0 => 1, 1 => 1);
    vV = ℤ₂Space(0 => 1);
    vL = ℤ₂Space(0 => 2, 1 => 1);
    vR = ℤ₂Space(0 => 2, 1 => 1);
end

# Pauli opertators
Id = Matrix{ComplexF64}(I,2,2);
X = 1/2 * [ 0 +1 ; +1 0 ];
Y = 1/2 * [ 0 -1im ; +1im 0 ];
Z = 1/2 * [ +1 0 ; 0 -1 ];

# generate the Ising MPO
ham_arr = zeros(Complex{Float64}, dim(vP), dim(vL), dim(vR), dim(vP));
ham_arr[:,1,1,:] = Id;
ham_arr[:,2,2,:] = Id;
ham_arr[:,1,3,:] = -J*X;
ham_arr[:,3,2,:] = X;
ham_arr[:,1,2,:] = -h*Z;
mpo = TensorMap(ham_arr, vP*vL, vR*vP)


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
ER_arr[2,1,1] = 1;
ER = TensorMap(ER_arr, vR*vV, vV);

groundStateEnergy = zeros(Float64, maxNumSteps, 5);

# initialize variables to be available outside of for-loop
eigenVal = 0;
prevEigenVal = 0;

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
        @tensor theta[-1 -2 -3; -4] := lambdaList[ia][-3 1] * gammaList[ib][-2 1 2] * gammaList[ic][-1 2 -4];
        # thetaInit = reshape(thetaInit,(chia*d*d*chic));
        # thetaInit = thetaInit ./ sqrt(abs(dot(thetaInit,thetaInit)));

        # function to apply the Hamiltonian to the wave function
        function applyH(X, EL, mpo, ER)
            @tensor Y[-1 -2 -3; -4] := EL[-3 3 1] * X[4 2 1 6] * mpo[-2 3 5 2] * mpo[-1 5 7 4] * ER[7 6 -4];
            return Y
        end
        
        # store previous eivenvalue
        prevEigenVal = eigenVal;
        prevEigenVec = theta;

        # optimize wave function
        eigenVal, eigenVec = let EL = EL, mpo = mpo, ER = ER
            eigsolve(theta,1, :SR, Lanczos()) do x
                applyH(x, EL, mpo, ER)
            end
        end
        gsEnergy = eigenVal[1];
        thetaP = eigenVec[1];
        
        #  perform SVD and truncate to desired bond dimension
        U, S, V, ϵ = tsvd(thetaP, (2,3), (1,4), trunc = truncdim(χ));
        V = permute(V, (2, 1), (3, ));

        # update environments
        function update_EL(EL, U, mpo)
            @tensor Y[-1; -2 -3] := EL[5 3 1] * U[2 1 -3] * mpo[4 3 -2 2] * conj(U[4 5 -1]);
            return Y
        end
        function update_ER(ER, V, mpo)
            @tensor Y[-1 -2; -3] := ER[3 1 5] * V[2 -2 1] * mpo[4 -1 3 2] * conj(V[4 -3 5]);
            return Y
        end
        global EL = update_EL(EL, U, mpo);
        global ER = update_ER(ER, V, mpo);

        # obtain the new tensors for MPS
        lambdaList[ib] = S;
        function guess(L1, U, L2)
            @tensor Y[-1 -2; -3] := inv(L1)[-2 1] * U[-1 1 2] * L2[2 -3];
            return Y
        end
        global gammaList[ib] = guess(lambdaList[ia], U, lambdaList[ib]);
        global gammaList[ic] = V;

        # print simulation progress
        @printf("%05i : E_iDMRG / Convergence / Discarded Weight / BondDim : %0.15f / %0.15f / %d \n",i,real(gsEnergy),ϵ,dim(space(S, 1)))

    end

end