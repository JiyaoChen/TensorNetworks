function extendSharedLink(mps1::DMRG_types.MPSTensor, mps2::DMRG_types.MPSTensor, EL::DMRG_types.MPOEnvLTensor, mpo1::DMRG_types.MPOTensor, mpo2::DMRG_types.MPOTensor, ER::DMRG_types.MPOEnvRTensor, χ::Int64, tol::Float64, dir::String)
    theta = mps1 * permute(mps2, (1,), (2,3))
    # get theta prime
    theta = applyH2(theta, EL, mpo1, mpo2, ER)  # bottleneck 2

    # perform SVD to obtain an optimal shared link dimension
    U, S, Vdag, ϵ = tsvd(theta, (1,2), (3,4), trunc = truncdim(min(χ)))    # bottleneck 1
    # U, S, Vdag, ϵ = tsvd(theta, (1,2), (3,4), trunc = truncerr(tol))    # bottleneck 1
    current_χ = dim(space(S,1))

    # set U and V to zero
    U = zero(U)
    Vdag = zero(Vdag)

    # extend mps1 & mps2 with the optimal variational subspace
    mps1 = catdomain(mps1,U)
    # mps1 = mps1 / sqrt(tr(mps1'*mps1))
    mps2 = permute(catcodomain(permute(mps2, (1,), (2,3)), Vdag), (1,2), (3,))
    # mps2 = mps2 / sqrt(tr(mps2'*mps2))

    # the environments must be updated
    if dir == "->"  # bottleneck 3
        L, Q = rightorth(mps2, (1,), (2,3), alg=TensorKit.LQpos())
        Q = permute(Q, (1,2), (3,))
        mps2 = Q
        mps1 = mps1 * L
        env = update_ER(ER, mps2, mpo2)
    else
        Q, R = leftorth(mps1, (1,2), (3,), alg=TensorKit.QRpos())
        mps1 = Q
        mps2 = permute(R * permute(mps2, (1,), (2,3)), (1,2), (3,))
        env = update_EL(EL, mps1, mpo1)
    end

    return mps1, mps2, env, current_χ, ϵ
end

function extendSharedLink2(mps1::DMRG_types.MPSTensor, mps2::DMRG_types.MPSTensor, EL::DMRG_types.MPOEnvLTensor, mpo1::DMRG_types.MPOTensor, mpo2::DMRG_types.MPOTensor, ER::DMRG_types.MPOEnvRTensor, χ::Int64)
    χMPS = dim(space(mps1,3))
    χP = dim(space(mps1,2))

    theta = mps1 * permute(mps2, (1,), (2,3))
    # get theta prime
    theta = applyH2(theta, EL, mpo1, mpo2, ER)  # bottleneck 2

    # perform SVD to obtain an optimal shared link dimension
    U, S, Vdag, ϵ = tsvd(theta, (1,2), (3,4), trunc = truncdim(min(χP*χMPS,χ)))    # bottleneck 1
    # U, S, Vdag, ϵ = tsvd(theta, (1,2), (3,4), trunc = truncerr(tol))    # bottleneck 1
    current_χ = dim(space(S,1))

    # set U and V to zero
    U = zero(U)
    Vdag = zero(Vdag)

    # extend mps1 & mps2 with the optimal variational subspace (filled with zeros)
    mps1 = catdomain(mps1, U)
    mps2 = permute(catcodomain(permute(mps2, (1,), (2,3)), Vdag), (1,2), (3,))

    # the environments must be updated
    env = update_ER(ER, mps2, mpo2)
    env = update_EL(EL, mps1, mpo1)

    return mps1, mps2, env, current_χ, ϵ
end

function extendSharedLink_fast(mps1::DMRG_types.MPSTensor, mps2::DMRG_types.MPSTensor, EL::DMRG_types.MPOEnvLTensor, mpo1::DMRG_types.MPOTensor, mpo2::DMRG_types.MPOTensor, ER::DMRG_types.MPOEnvRTensor, χ::Int64, tol::Float64, dir::String)
    v1 = fuse(space(mps1,1),space(mps1,2))
    v2 = fuse(space(mps2,2)',space(mps2,3)')
    
    vM = infimum(v1, v2).dims  # create dict from the overlap of v1 and v2
    vM.values .= 1  # setting all degeneracies to 1
    vM = typeof(v1)(vM)  # create a new space with truncated degeneracies

    vM = space(mps1,3)' ⊗ vM  # concatenate the link dimension
    println(vM)

    # set U and V to zero
    U = TensorMap(zeros, codomain(mps1), vM)
    Vdag = TensorMap(zeros, vM, domain(permute(mps2,(1,), (2,3))))

    # extend mps1 & mps2 with the optimal variational subspace (initially filled with zeros)
    mps1 = catdomain(mps1, U)
    mps2 = permute(catcodomain(permute(mps2, (1,), (2,3)), Vdag), (1,2), (3,))

    # after the previous step, one of the environments needs to be updated
    if dir == "->"  # bottleneck 3
        env = update_ER(ER, mps2, mpo1)
    else
        env = update_EL(EL, mps1, mpo1)
    end

    return mps1, mps2, env
end