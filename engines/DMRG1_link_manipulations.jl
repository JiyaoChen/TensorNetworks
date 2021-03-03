function extendSharedLink(mps1::DMRG_types.MPSTensor, mps2::DMRG_types.MPSTensor, EL::DMRG_types.MPOEnvLTensor, mpo1::DMRG_types.MPOTensor, mpo2::DMRG_types.MPOTensor, ER::DMRG_types.MPOEnvRTensor, χ::Int64, dir::String)
    theta = mps1 * permute(mps2, (1,), (2,3))
    # get theta prime
    theta = applyH2(theta, EL, mpo1, mpo2, ER)  # bottleneck 2

    # perform SVD to obtain an optimal shared link dimension
    U, S, Vdag, ϵ = tsvd(theta, (1,2), (3,4), trunc = truncdim(χ))    # bottleneck 1
    current_χ = dim(space(S,1))

    # set U and V to zero
    U = zero(U)
    Vdag = zero(Vdag)

    # extend mps1 & mps2 with the optimal variational subspace (initially filled with zeros)
    mps1 = catdomain(mps1, U)
    mps2 = permute(catcodomain(permute(mps2, (1,), (2,3)), Vdag), (1,2), (3,))

    # after the previous step, one of the environments needs to be updated
    if dir == "->"  # bottleneck 3
        env = update_ER(ER, mps2, mpo1)
    else
        env = update_EL(EL, mps1, mpo1)
    end

    return mps1, mps2, env, current_χ, ϵ
end