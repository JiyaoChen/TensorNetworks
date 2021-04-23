using KrylovKit, LinearAlgebra, TensorOperations

"""
    applyH1(AC, FL, FR, M)
Apply the effective Hamiltonian on the center tensor `AC`, by contracting with the left and right
environment `FL` and `FR` and the MPO tensor `M`
"""
function applyH1(AC, FL, FR, MPO)
    @tensor HAC[-1 -2; -3] := FL[-1 1 3] * AC[3 2 5] * MPO[1 -2 4 2] * FR[5 4 -3];
end

"""
    applyH0(C, FL, FR)
Apply the effective Hamiltonian on the bond matrix C, by contracting with the left and right
environment `FL` and `FR`
"""
function applyH0(C, FL, FR)
    @tensor HC[-1; -2] := FL[-1 3 1] * C[1 2] * FR[2 3 -2];
end

"""
    leftenv(A, M, FL; kwargs)
Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of A - M - conj(A) contracted along the physical dimension.
"""
function leftEnv(MPS, MPO, FL = TensorMap(randn, ComplexF64, space(MPS,1), space(MPO,1) ⊗ space(MPS,1)); kwargs...)
    
    λs, FLs, info = eigsolve(FL, 1, :LM; ishermitian = false, kwargs...) do FL
        @tensor FL[-1; -2 -3] := FL[1 3 5] * MPS[5 4 -3] * MPO[3 2 -2 4] * conj(MPS[1 2 -1]);
    end

    return FLs[1], real(λs[1]), info
end
"""
    rightenv(A, M, FR; kwargs...)
Compute the right environment tensor for MPS A and MPO M, by finding the right fixed point
of A - M - conj(A) contracted along the physical dimension.
"""
function rightEnv(MPS, MPO, FR = TensorMap(randn, ComplexF64, space(MPS,1) ⊗ space(MPO,1), space(MPS,1)); kwargs...)

    λs, FRs, info = eigsolve(FR, 1, :LM; ishermitian = false, kwargs...) do FR
        @tensor FR[-1 -2; -3] := MPS[-1 4 5] * MPO[-2 2 3 4] * conj(MPS[-3 2 1]) * FR[5 3 1];
    end
    return FRs[1], real(λs[1]), info
end

function optimalSubSpaceExpansion(AL, C, AR, MPO, FL, FR, chiE, tol::Float64; kwargs...)

    # construct tensor A2C
    @tensor AC[-1 -2; -3] := AL[-1 -2 1] * C[1 -3];
    @tensor A2C[-1 -2; -3 -4] := FL[-1 1 3] * AC[3 2 5] * AR[5 7 8] * MPO[1 -2 4 2] * MPO[4 -3 6 7] * FR[8 6 -4];

    # construct nullspaces for AL and AR
    NL = leftnull(AL, (1,2), (3,));
    NR = rightnull(AR, (1,), (2,3));

    # determine optimal expansion space
    newA2C = adjoint(NL) * A2C * adjoint(NR)
    U, S, V = tsvd(newA2C, (1,), (2,), trunc = truncdim(chiE));

    expL = permute(NL * U, (1,2), (3,));
    expR = permute(V * NR, (1,), (2,3));

    # do optimal expansion and update AL, AC and AR
    tempAL = permute(catdomain(permute(AL, (1,2), (3,)), expL), (1,), (2,3))
    lz = TensorMap(zeros, ComplexF64, space(expL, 3)', domain(tempAL))
    AL = permute(catcodomain(tempAL, lz), (1,2), (3,));

    tempAR = permute(catcodomain(permute(AR, (1,), (2,3)), expR), (1,2), (3,))
    rz = TensorMap(zeros, ComplexF64, codomain(tempAR), space(expR, 1))
    AR = permute(catdomain(tempAR, rz), (1,2), (3,));

    l = TensorMap(zeros, ComplexF64, codomain(C), space(expR, 1))
    C = catdomain(C, l)
    r = TensorMap(zeros, ComplexF64, space(expL, 3)', domain(C))
    C = catcodomain(C, r)
    AC = AL * C;

    # # update environments
    # FL, = leftEnv(AL, MPO, FL; tol = tol/10, kwargs...)
    # FR, = rightEnv(AR, MPO, FR; tol = tol/10, kwargs...)

    # function return
    return AL, C, AR, FL, FR

end

function vumps(MPS, MPO, chiE; verbose = true, tol = 1e-6, kwargs...)
    
    AL, = leftorth(MPS,(1,2),(3,),alg = QRpos())
    C, AR = rightorth(AL,(1,),(2,3),alg = LQpos())
    AR = permute(AR, (1,2), (3,))

    FL, λL = leftEnv(AL, MPO; kwargs...)
    FR, λR = rightEnv(AR, MPO; kwargs...)

    verbose && println("Starting point has λ ≈ $λL ≈ $λR")

    # make optimization step
    λ, AL, C, AR, = vumpsstep(AL, C, AR, MPO, FL, FR; tol = tol)
    # AL, C, = leftorth(AR, C; tol = tol/10, kwargs...) # regauge MPS: not really necessary

    # make an optimal subspace expansion
    AL, C, AR, FL, FR = optimalSubSpaceExpansion(AL, C, AR, MPO, FL, FR, chiE, tol);

    # compute environments
    # FL, λL = leftEnv(AL, MPO, FL; tol = tol/10, kwargs...)
    # FR, λR = rightEnv(AR, MPO, FR; tol = tol/10, kwargs...)

    FL, λL = leftEnv(AL, MPO; kwargs...)
    FR, λR = rightEnv(AR, MPO; kwargs...)
    envNorm = @tensor FL[1 2 3] * C[3 4] * conj(C[1 5]) * FR[4 2 5]
    FL /= envNorm   # normalize FL, not really necessary
    FR /= envNorm   # normalize FR, not really necessary

    # Convergence measure: norm of the projection of the residual onto the tangent space
    @tensor AC[-1 -2; -3] := AL[-1 -2 1] * C[1 -3];
    MAC = applyH1(AC, FL, FR, MPO)
    @tensor MAC[a,s,b] -= AL[a,s,b'] * (conj(AL[a',s',b']) * MAC[a',s',b])
    err = norm(MAC)
    i = 1
    verbose && @printf("Step %d: λL ≈ %0.15f, λR ≈ %0.15f, err ≈ %0.15f\n", i, λL, λR, err)
    while err > tol
        tol = minimum([tol, err/10])
        # make optimization step
        λ, AL, C, AR, = vumpsstep(AL, C, AR, MPO, FL, FR; tol = tol)
        # AL, C, = leftorth(AR, C; tol = tol/10, kwargs...) # regauge MPS: not really necessary

        # make an optimal subspace expansion
        @time AL, C, AR, FL, FR = optimalSubSpaceExpansion(AL, C, AR, MPO, FL, FR, chiE, tol);

        # compute environments
        # FL, λL = leftEnv(AL, MPO, FL; tol = tol/10, kwargs...)
        # FR, λR = rightEnv(AR, MPO, FR; tol = tol/10, kwargs...)
        FL, λL = leftEnv(AL, MPO; kwargs...)
        FR, λR = rightEnv(AR, MPO; kwargs...)
        
        # normalize FL and FR, not really necessary
        envNorm = @tensor FL[1 2 3] * C[3 4] * conj(C[1 5]) * FR[4 2 5]
        FL /= envNorm
        FR /= envNorm

        # Convergence measure: norm of the projection of the residual onto the tangent space
        @tensor AC[-1 -2; -3] := AL[-1 -2 1] * C[1 -3];
        MAC = applyH1(AC, FL, FR, MPO)
        @tensor MAC[a,s,b] -= AL[a,s,b'] * (conj(AL[a',s',b']) * MAC[a',s',b])
        err = norm(MAC)
        i += 1
        verbose && @printf("Step %d: λL ≈ %0.15f, λR ≈ %0.15f, err ≈ %0.15f\n", i, λL, λR, err)
    end

    return λ, AL, C, AR, FL, FR
end

"""
    function vumpsstep(AL, C, AR, FL, FR; kwargs...)
Perform one step of the VUMPS algorithm
"""
function vumpsstep(AL, C, AR, MPO, FL, FR; kwargs...)
    
    @tensor AC[-1 -2; -3] := AL[-1 -2 1] * C[1 -3];
    μ1s, ACs, info1 = eigsolve(x -> applyH1(x, FL, FR, MPO), AC, 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
    μ0s, Cs, info0 = eigsolve(x -> applyH0(x, FL, FR), C, 1; ishermitian = false, maxiter = 1, kwargs...)
    λ = real(μ1s[1]/μ0s[1])
    AC = ACs[1]
    C = Cs[1]

    QAC, RAC = leftorth(AC, (1,2), (3,), alg = QRpos())
    QC, RC = leftorth(C, (1,), (2,), alg = QRpos())
    AL = permute(QAC * QC', (1,2), (3,))
    errL = norm(RAC - RC)

    LAC, QAC = rightorth(AC, (1,), (2,3), alg = LQpos())
    LC, QC = rightorth(C, (1,), (2,), alg = LQpos())
    AR = permute(QC' * QAC, (1,2), (3,));
    errR = norm(LAC - LC)

    return λ, AL, C, AR, errL, errR
end