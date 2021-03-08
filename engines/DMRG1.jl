# collection of necessary contractions
include("DMRG_contractions.jl")
include("DMRG1_link_manipulations.jl")

function verbosePrintDMRG1(s::Int64,i::Int64,currEigenVal::Float64,ϵ::Float64,tol::Float64,current_χ::Int64,time::Float64)
    @info("DMRG1 -- Sweep "*string(s),i,currEigenVal,ϵ,tol,current_χ,time)
end

function DMRG1(mps::DMRG_types.MPS, env::DMRG_types.MPOEnvironments, model::DMRG_types.Model)
    
    # get parameters from dict
    χ = model.P["χ"]
    maxiter = model.P["maxiter"]
    krylovdim = model.P["krylovdim"]
    solver = model.P["solver"]

    current_χ = Array{Int64}(undef, length(mps.ARs) - 1)
    ϵ = Array{Float64}(undef, length(mps.ARs))
    tol = Array{Float64}(undef, length(mps.ARs))
    tol .= model.P["tol"] 

    mintol = model.P["tol"] 

    # DMRG sweeps
    for s = 1 : length(χ)
        
        # link extension
        # if maximum(current_χ) != χ[s]
            # println("blow up link dimension")
            for i = 1 : length(mps.ARs) - 1
                (mps.ACs[i], mps.ARs[i+1], env.mpoEnvR[i], current_χ[i], ϵ[i]) = extendSharedLink(mps.ACs[i], mps.ARs[i+1], env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1], χ[s], tol[i], "->")
                (Q, R) = leftorth(mps.ACs[i], (1,2), (3,), alg=TensorKit.QRpos())
                mps.ALs[i] = Q
                mps.ACs[i+1] = permute(R * permute(mps.ARs[i+1], (1,), (2,3)), (1,2), (3,))
                env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], mps.ALs[i], model.H.mpo[i])
            end
            for i = length(mps.ARs) : -1 : 2
                # (mps.ALs[i-1], mps.ACs[i], env.mpoEnvL[i], current_χ[i], ϵ[i]) = extendSharedLink(mps.ALs[i-1], mps.ACs[i], env.mpoEnvL[i-1], model.H.mpo[i-1], model.H.mpo[i], env.mpoEnvR[i], χ[s], tol[i], "<-")
                (L, Q) = rightorth(mps.ACs[i], (1,), (2,3), alg=TensorKit.LQpos())
                mps.ARs[i] = permute(Q, (1,2), (3,))
                mps.ACs[i-1] = mps.ALs[i-1] * L
                env.mpoEnvR[i-1] = update_ER(env.mpoEnvR[i], mps.ARs[i], model.H.mpo[i])
            end
        # end

        # left -> right sweeping
        for i = 1 : length(mps.ARs) - 1

            # construct initial theta
            theta = permute(mps.ACs[i], (1,2,3), ())

            # optimize wave function
            elapsedTime = @elapsed eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, solver(tol=tol[i],maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH1(x, env.mpoEnvL[i], model.H.mpo[i], env.mpoEnvR[i])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3] * conj(currEigenVec[1 2 3])
            tol[i] = min(mintol,abs(overlap))

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

            verbosePrintDMRG1(s,i,real(currEigenVal),ϵ[i],tol[i],current_χ[i],elapsedTime)
            
        end

        # right -> left sweeping
        for i = length(mps.ARs) : -1 : 2
            
            # construct initial theta
            theta = permute(mps.ACs[i], (1,2,3), ())

            # optimize wave function
            elapsedTime = @elapsed eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, solver(tol=tol[i],maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH1(x, env.mpoEnvL[i], model.H.mpo[i], env.mpoEnvR[i])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3] * conj(currEigenVec[1 2 3])
            tol[i] = min(mintol,abs(overlap))

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

            verbosePrintDMRG1(s,i,real(currEigenVal),ϵ[i],tol[i],current_χ[i-1],elapsedTime)
            
        end

        if maximum(tol) < 1e-14 && maximum(current_χ) == maximum(χ)
            return mps
        end
    end

    return mps
end