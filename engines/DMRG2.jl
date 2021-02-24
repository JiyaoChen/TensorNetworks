function DMRG2(mps::DMRG_types.MPS, env::DMRG_types.MPOEnvironments, model::DMRG_types.Model)
    
    # get parameters from dict
    χ = model.P["χ"]
    tol = model.P["tol"]
    sweeps = model.P["sweeps"]
    maxiter = model.P["maxiter"]
    krylovdim = model.P["krylovdim"]
    solver = model.P["solver"]

    for s = 1 : sweeps
        # left -> right sweeping
        for i = 1 : length(mps.ARs) - 1
            # construct initial theta
            theta = permute(mps.ACs[i] * permute(mps.ARs[i+1], (1,), (2,3)), (1,2,3), (4,))

            # optimize wave function
            eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, Lanczos(tol=tol,maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH(x, env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3 4] * conj(currEigenVec[1 2 3 4])
            tol = abs(overlap)

            #  perform SVD and truncate to desired bond dimension
            U, S, Vdag, ϵ = tsvd(currEigenVec, (1,2), (3,4), trunc = truncdim(χ))
            current_χ = dim(space(S,1))
            Vdag = permute(Vdag, (1,2), (3,))

            # construct new AC
            @tensor mps.ACs[i+1][-1 -2; -3] := S[-1 1] * Vdag[1 -2 -3];

            # update tensors in mps
            mps.ALs[i] = U
            # mps.ARs[i+1] = Vdag

            # update left environments
            env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], U, model.H.mpo[i])
            # env.mpoEnvR[i] = update_ER(env.mpoEnvR[i+1], Vdag, model.H.mpo[i+1])

            @printf("%03i/%03i : E_DMRG2 / Discarded Weight / tol / BondDim : %0.15f / %0.15f / %0.15f / %d \n",s,i,real(currEigenVal),ϵ,tol,current_χ)
            
        end

        # right -> left sweeping
        for i = length(mps.ARs) - 1 : -1 : 1
            # construct initial theta
            theta = permute(mps.ALs[i] * permute(mps.ACs[i+1], (1,), (2,3)), (1,2,3), (4,))

            # optimize wave function
            eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, Lanczos(tol=tol,maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH(x, env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3 4] * conj(currEigenVec[1 2 3 4])
            tol = abs(overlap)

            #  perform SVD and truncate to desired bond dimension
            U, S, Vdag, ϵ = tsvd(currEigenVec, (1,2), (3,4), trunc = truncdim(χ))
            current_χ = dim(space(S,1))
            Vdag = permute(Vdag, (1,2), (3,))

            # update tensors in mps
            # mps.ALs[i] = U
            mps.ARs[i+1] = Vdag

            # construct new AC
            mps.ACs[i] = U * S;

            # update left environments
            # env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], U, model.H.mpo[i])
            env.mpoEnvR[i] = update_ER(env.mpoEnvR[i+1], Vdag, model.H.mpo[i+1])

            @printf("%03i/%03i : E_DMRG2 / Discarded Weight / tol / BondDim : %0.15f / %0.15f / %0.15f / %d \n",s,i,real(currEigenVal),ϵ,tol,current_χ)
            
        end
    end

end