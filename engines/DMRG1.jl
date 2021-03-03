# collection of necessary contractions
include("DMRG_contractions.jl")
include("DMRG1_link_manipulations.jl")

function verbosePrint(s::Int64,i::Int64,currEigenVal::Float64,ϵ::Float64,tol::Float64,current_χ::Int64)
    @info("DMRG1 -- Sweep "*string(s),i,currEigenVal,ϵ,tol,current_χ)
end

function DMRG1(mps::DMRG_types.MPS, env::DMRG_types.MPOEnvironments, model::DMRG_types.Model)
    
    # get parameters from dict
    χ = model.P["χ"]
    tol = model.P["tol"]
    maxiter = model.P["maxiter"]
    krylovdim = model.P["krylovdim"]
    solver = model.P["solver"]

    current_χ = 0
    ϵ = 0.0

    for s = 1 : length(χ)
        # left -> right sweeping
        for i = 1 : length(mps.ARs) - 1
            (mps.ACs[i], mps.ARs[i+1], env.mpoEnvR[i], current_χ, ϵ) = extendSharedLink(mps.ACs[i], mps.ARs[i+1], env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1], χ[s], tol, "->")
            # (mps.ACs[i], mps.ARs[i+1], env.mpoEnvR[i]) = extendSharedLink_fast(mps.ACs[i], mps.ARs[i+1], env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1], χ[s], tol, "->")
            
            # construct initial theta
            theta = mps.ACs[i]

            # optimize wave function
            eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, Lanczos(tol=tol,maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH1(x, env.mpoEnvL[i], model.H.mpo[i], env.mpoEnvR[i])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3] * conj(currEigenVec[1 2 3])
            tol = abs(overlap)

            # bring updated tensor in left canonical form
            (Q, R) = leftorth(currEigenVec, (1,2), (3,), alg=TensorKit.QRpos())
            # Q, S, R, ϵ = tsvd(currEigenVec, (1,2), (3,), trunc = truncdim(χ[s]))
            # R = S * R

            # construct new AC
            mps.ACs[i+1] = permute(R * permute(mps.ARs[i+1], (1,), (2,3)), (1,2), (3,))

            # update tensors in mps
            mps.ALs[i] = Q

            # update left environments
            env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], mps.ALs[i], model.H.mpo[i])
            # env.mpoEnvR[i] = update_ER(env.mpoEnvR[i+1], Vdag, model.H.mpo[i+1])

            verbosePrint(s,i,real(currEigenVal),ϵ,tol,current_χ)
            
        end

        # right -> left sweeping
        for i = length(mps.ARs) : -1 : 2
            (mps.ALs[i-1], mps.ACs[i], env.mpoEnvL[i], current_χ, ϵ) = extendSharedLink(mps.ALs[i-1], mps.ACs[i], env.mpoEnvL[i-1], model.H.mpo[i-1], model.H.mpo[i], env.mpoEnvR[i], χ[s], tol, "<-")
            # (mps.ALs[i-1], mps.ACs[i], env.mpoEnvL[i]) = extendSharedLink_fast(mps.ALs[i-1], mps.ACs[i], env.mpoEnvL[i-1], model.H.mpo[i-1], model.H.mpo[i], env.mpoEnvR[i], χ[s], tol, "<-")
            
            # construct initial theta
            theta = mps.ACs[i]

            # optimize wave function
            eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, Lanczos(tol=tol,maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH1(x, env.mpoEnvL[i], model.H.mpo[i], env.mpoEnvR[i])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3] * conj(currEigenVec[1 2 3])
            tol = abs(overlap)

            # println(currEigenVec)
            # bring updated tensor in right canonical form
            (L, Q) = rightorth(currEigenVec, (1,), (2,3), alg=TensorKit.LQpos())
            # L, S, Q, ϵ = tsvd(currEigenVec, (1,), (2,3), trunc = truncdim(χ[s]))
            # L = L * S
            Q = permute(Q, (1,2), (3,))

            # update tensors in mps
            mps.ARs[i] = Q

            # construct new AC
            mps.ACs[i-1] = mps.ALs[i-1] * L

            # update left environments
            # env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], U, model.H.mpo[i])
            env.mpoEnvR[i-1] = update_ER(env.mpoEnvR[i], mps.ARs[i], model.H.mpo[i])

            verbosePrint(s,i,real(currEigenVal),ϵ,tol,current_χ)
            
        end
    end

    return mps
end