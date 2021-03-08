# collection of necessary contractions
include("DMRG_contractions.jl")

function verbosePrintDMRG2(s::Int64,i::Int64,currEigenVal::Float64,ϵ::Float64,tol::Float64,current_χ::Int64,time::Float64)
    @info("DMRG2 -- Sweep "*string(s),i,currEigenVal,ϵ,tol,current_χ,time)
end

function DMRG2(mps::DMRG_types.MPS, env::DMRG_types.MPOEnvironments, model::DMRG_types.Model)
    
    # get parameters from dict
    χ = model.P["χ"]
    maxiter = model.P["maxiter"]
    krylovdim = model.P["krylovdim"]
    solver = model.P["solver"]

    current_χ = Array{Int64}(undef, length(mps.ARs)-1)
    ϵ = Array{Float64}(undef, length(mps.ARs)-1)
    tol = Array{Float64}(undef, length(mps.ARs)-1)
    tol .= model.P["tol"] 

    for s = 1 : length(χ)
        # left -> right sweeping
        for i = 1 : length(mps.ARs) - 1
            # construct initial theta
            theta = permute(mps.ACs[i] * permute(mps.ARs[i+1], (1,), (2,3)), (1,2,3,4), ())

            # optimize wave function
            elapsedTime = @elapsed eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, solver(tol=tol[i],maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH2(x, env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3 4] * conj(currEigenVec[1 2 3 4])
            tol[i] = abs(overlap)

            #  perform SVD and truncate to desired bond dimension
            U, S, Vdag, ϵ[i] = tsvd(currEigenVec, (1,2), (3,4), trunc = truncdim(χ[s]))
            current_χ[i] = dim(space(S,1))
            Vdag = permute(Vdag, (1,2), (3,))

            # construct new AC
            @tensor mps.ACs[i+1][-1 -2; -3] := S[-1 1] * Vdag[1 -2 -3];

            # update tensors in mps
            mps.ALs[i] = U
            # mps.ARs[i+1] = Vdag

            # update left environments
            env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], U, model.H.mpo[i])
            # env.mpoEnvR[i] = update_ER(env.mpoEnvR[i+1], Vdag, model.H.mpo[i+1])

            verbosePrintDMRG2(s,i,real(currEigenVal),ϵ[i],tol[i],current_χ[i],elapsedTime)
            
        end

        # right -> left sweeping
        for i = length(mps.ARs) - 1 : -1 : 1
            # construct initial theta
            theta = permute(mps.ALs[i] * permute(mps.ACs[i+1], (1,), (2,3)), (1,2,3,4), ())

            # optimize wave function
            elapsedTime = @elapsed eigenVal, eigenVec = 
                eigsolve(theta,1, :SR, solver(tol=tol[i],maxiter=maxiter,krylovdim=krylovdim)) do x
                    applyH2(x, env.mpoEnvL[i], model.H.mpo[i], model.H.mpo[i+1], env.mpoEnvR[i+1])
                end
            currEigenVal = eigenVal[1]
            currEigenVec = eigenVec[1]

            # computes the overlap
            overlap = 1 - @tensor theta[1 2 3 4] * conj(currEigenVec[1 2 3 4])
            tol[i] = abs(overlap)

            #  perform SVD and truncate to desired bond dimension
            U, S, Vdag, ϵ[i] = tsvd(currEigenVec, (1,2), (3,4), trunc = truncdim(χ[s]))
            current_χ[i] = dim(space(S,1))
            Vdag = permute(Vdag, (1,2), (3,))

            # update tensors in mps
            # mps.ALs[i] = U
            mps.ARs[i+1] = Vdag

            # construct new AC
            mps.ACs[i] = U * S;

            # update left environments
            # env.mpoEnvL[i+1] = update_EL(env.mpoEnvL[i], U, model.H.mpo[i])
            env.mpoEnvR[i] = update_ER(env.mpoEnvR[i+1], Vdag, model.H.mpo[i+1])

            verbosePrintDMRG2(s,i,real(currEigenVal),ϵ[i],tol[i],current_χ[i],elapsedTime)
            
        end

        println(maximum(tol))
        println(maximum(current_χ))
        if maximum(tol) < 1e-14 && maximum(current_χ) == maximum(χ)
            return mps
        end
    end

    return mps
end