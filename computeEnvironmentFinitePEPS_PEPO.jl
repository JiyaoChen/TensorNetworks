include("environments_finitePEPS.jl")

function computeEnvironmentFinitePEPS_PEPO(finitePEPS, finitePEPO, chiE)

    # get size of finitePEPS
    Lx, Ly = size(finitePEPS);

    # initialize CTM tensors around the finitePEPS circumference
    envTensors = initializeEnvironmentTensors(finitePEPS, finitePEPO);

    # compute effective environments using boundary MPS methods from above
    rowEnvironments_U = Array{Array{Any,1},1}(undef, Lx);
    for idx = 1 : Lx
        
        # construct boundaryMPS
        boundaryMPS = Array{Any,1}(undef, Ly + 2);
        if idx == 1

            boundaryMPS[1] = envTensors[idx,1][1];
            for idy = 1 : Ly
                boundaryMPS[idy + 1] = envTensors[idx,idy][2];
            end
            boundaryMPS[Ly + 2] = envTensors[idx,Ly][3];

        else

            @tensor boundaryMPS[1][-1; -2 -3 -4 -5] := envTensors[idx - 1,1][8][-1 -2 -3 -4 1] * rowEnvironments_U[idx - 1][1][1 -5];
            for idy = 1 : Ly
                @tensor boundaryMPS[idy + 1][-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11] := finitePEPS[idx - 1,idy][-2 -5 4 -10 5] * finitePEPO[idx - 1,idy][-3 -6 2 -9 3 4] * conj(finitePEPS[idx - 1,idy][-4 -7 2 -8 1]) * rowEnvironments_U[idx - 1][idy + 1][-1 5 3 1 -11];
            end
            @tensor boundaryMPS[Ly + 2][-1 -2 -3 -4 -5; ()] := envTensors[idx - 1,Ly][4][-2 -3 -4 -5 1] * rowEnvironments_U[idx - 1][Ly + 2][-1 1];
            
            # use SVD to compress boundaryMPS
            for idy = 1 : Ly + 1

                if idy == 1
                    @tensor twoSite[-1 -2 -3 -4; -5 -6 -7 -8] := boundaryMPS[idy][-1 1 2 3 4] * boundaryMPS[idy + 1][4 3 2 1 -2 -3 -4 -5 -6 -7 -8];
                    U, S, V, ϵ = tsvd(twoSite, (1,), (2,3,4,5,6,7,8), trunc = truncdim(chiE));
                    @tensor newU[-1; -2] := U[-1 1] * sqrt(S)[1 -2];
                    @tensor newV[-1 -2 -3 -4; -5 -6 -7 -8] := sqrt(S)[-1 1] * permute(V,(1,2,3,4), (5,6,7,8))[1 -2 -3 -4 -5 -6 -7 -8];
                elseif idy == Ly + 1
                    @tensor twoSite[-1 -2 -3 -4 -5; ()] := boundaryMPS[idy][-1 -2 -3 -4 1 2 3 4] * boundaryMPS[idy + 1][4 3 2 1 -5];
                    U, S, V, ϵ = tsvd(twoSite, (1,2,3,4), (5,), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3 -4; -5] := U[-1 -2 -3 -4 1] * sqrt(S)[1 -5];
                    @tensor newV[-1 -2; ()] := sqrt(S)[-1 1] * permute(V,(1,2), ())[1 -2];
                else
                    @tensor twoSite[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11] := boundaryMPS[idy][-1 -2 -3 -4 1 2 3 4] * boundaryMPS[idy + 1][4 3 2 1 -5 -6 -7 -8 -9 -10 -11];
                    U, S, V, ϵ = tsvd(twoSite, (1,2,3,4), (5,6,7,8,9,10,11), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3 -4; -5] := U[-1 -2 -3 -4 1] * sqrt(S)[1 -5];
                    @tensor newV[-1 -2 -3 -4; -5 -6 -7 -8] := sqrt(S)[-1 1] * permute(V,(1,2,3,4), (5,6,7,8))[1 -2 -3 -4 -5 -6 -7 -8];
                end

                boundaryMPS[idy] = newU;
                boundaryMPS[idy + 1] = newV;

            end

        end

        # store boundaryMPS
        rowEnvironments_U[idx] = boundaryMPS;

    end

    # compute effective environments using boundary MPS methods from below
    rowEnvironments_D = Array{Array{Any,1},1}(undef, Lx);
    for idx = Lx : -1 : 1

        # construct boundaryMPS
        boundaryMPS = Array{Any,1}(undef, Ly + 2);
        if idx == Lx

            boundaryMPS[1] = envTensors[idx,1][7];
            for idy = 1 : Ly
                boundaryMPS[idy + 1] = envTensors[idx,idy][6];
            end
            boundaryMPS[Ly + 2] = envTensors[idx,Ly][5];

        else

            @tensor boundaryMPS[1][(); -1 -2 -3 -4 -5] := rowEnvironments_D[idx + 1][1][1 -1] * envTensors[idx + 1,1][8][1 -2 -3 -4 -5];
            for idy = 1 : Ly
                @tensor boundaryMPS[idy + 1][-1 -2 -3 -4; -5 -6 -7 -8 -9 -10 -11] := rowEnvironments_D[idx + 1][idy + 1][-4 -5 1 3 5] * finitePEPS[idx + 1,idy][-1 5 4 -8 -11] * finitePEPO[idx + 1,idy][-2 3 2 -7 -10 4] * conj(finitePEPS[idx + 1,idy][-3 1 2 -6 -9]);
            end
            @tensor boundaryMPS[Ly + 2][-1 -2 -3 -4; -5] := rowEnvironments_D[idx + 1][Ly + 2][-4 1] * envTensors[idx + 1,Ly][4][-1 -2 -3 1 -5];

            # use SVD to compress boundaryMPS
            for idy = Ly + 2 : -1 : 2

                if idy == Ly + 2
                    @tensor twoSite[-1 -2 -3 -4; -5 -6 -7 -8] := boundaryMPS[idy - 1][-1 -2 -3 -4 1 2 3 4 -6 -7 -8] * boundaryMPS[idy][4 3 2 1 -5];
                    U, S, V, ϵ = tsvd(twoSite, (6,7,8,1,2,3,4), (5,), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3 -4; -5 -6 -7 -8] := permute(U, (4,5,6,7), (8,1,2,3))[-1 -2 -3 -4 1 -6 -7 -8] * sqrt(S)[1 -5];
                    @tensor newV[-1; -2] := sqrt(S)[-1 1] * V[1 -2];
                elseif idy == 2
                    @tensor twoSite[(); -1 -2 -3 -4 -5] := boundaryMPS[idy - 1][1 2 3 4 -5] * boundaryMPS[idy][4 3 2 1 -1 -2 -3 -4];
                    U, S, V, ϵ = tsvd(twoSite, (5,), (1,2,3,4), trunc = truncdim(chiE));
                    @tensor newU[(); -1 -2] := permute(U, (), (2,1))[1 -2] * sqrt(S)[1 -1];
                    @tensor newV[-1; -2 -3 -4 -5] := sqrt(S)[-1 1] * V[1 -2 -3 -4 -5];
                else
                    @tensor twoSite[-1 -2 -3 -4; -5 -6 -7 -8 -9 -10 -11] := boundaryMPS[idy - 1][-1 -2 -3 -4 1 2 3 4 -9 -10 -11] * boundaryMPS[idy][4 3 2 1 -5 -6 -7 -8];
                    U, S, V, ϵ = tsvd(twoSite, (9,10,11,1,2,3,4), (5,6,7,8), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3 -4; -5 -6 -7 -8] := permute(U, (4,5,6,7), (8,1,2,3))[-1 -2 -3 -4 1 -6 -7 -8] * sqrt(S)[1 -5];
                    @tensor newV[-1; -2 -3 -4 -5] := sqrt(S)[-1 1] * V[1 -2 -3 -4 -5];
                end

                boundaryMPS[idy - 1] = newU;
                boundaryMPS[idy] = newV;
                
            end

        end

        # store boundaryMPS
        rowEnvironments_D[idx] = boundaryMPS;

    end

    # compute and assign all CTM tensors
    for idx = 1 : Lx

        # perform left to right contraction
        colEnvironments_L = Array{Array{Any,1},1}(undef, Ly);
        for idy = 1 : Ly

            # construct boundaryMPS
            boundaryMPS = Array{Any,1}(undef,3);
            if idy == 1

                boundaryMPS[1] = rowEnvironments_U[idx][1];
                boundaryMPS[2] = envTensors[idx,idy][8];
                boundaryMPS[3] = rowEnvironments_D[idx][1];

            else

                @tensor boundaryMPS[1][-1 -2 -3 -4; -5] := colEnvironments_L[idy - 1][1][-1 1] * rowEnvironments_U[idx][idy][1 -2 -3 -4 -5];
                @tensor boundaryMPS[2][-1 -2 -3 -4; -5 -6 -7 -8 -9 -10 -11] := colEnvironments_L[idy - 1][2][-1 1 3 5 -11] * finitePEPS[idx,idy - 1][5 -2 4 -7 -10] * finitePEPO[idx,idy - 1][3 -3 2 -6 -9 4] * conj(finitePEPS[idx,idy - 1][1 -4 2 -5 -8]);
                @tensor boundaryMPS[3][(); -1 -2 -3 -4 -5] := colEnvironments_L[idy - 1][3][1 -5] * rowEnvironments_D[idx][idy][1 -1 -2 -3 -4];

                # use SVD to compress boundaryMPS
                @tensor twoSite[-1 -2 -3 -4; -5 -6 -7 -8] := boundaryMPS[2][-1 -2 -3 -4 -5 -6 -7 1 2 3 4] * boundaryMPS[1][4 3 2 1 -8];
                U, S, V, ϵ = tsvd(twoSite, (1,2,3,4,5,6,7), (8,), trunc = truncdim(chiE));
                @tensor newU[-1 -2 -3 -4; -5 -6 -7 -8] := permute(U, (1,2,3,4), (5,6,7,8))[-1 -2 -3 -4 -5 -6 -7 1] * sqrt(S)[1 -8];
                @tensor newV[-1; -2] := sqrt(S)[-1 1] * V[1 -2];
                boundaryMPS[1] = newV;
                boundaryMPS[2] = newU;

                @tensor twoSite[(); -1 -2 -3 -4 -5] := boundaryMPS[3][-1 1 2 3 4] * boundaryMPS[2][4 3 2 1 -2 -3 -4 -5];
                U, S, V, ϵ = tsvd(twoSite, (1,), (2,3,4,5), trunc = truncdim(chiE));
                @tensor newU[(); -1 -2] := permute(U, (), (1,2))[-1 1] * sqrt(S)[1 -2];
                @tensor newV[-1; -2 -3 -4 -5] := sqrt(S)[-1 1] * V[1 -2 -3 -4 -5];
                boundaryMPS[2] = newV;
                boundaryMPS[3] = newU;

            end

            # store boundaryMPS
            colEnvironments_L[idy] = boundaryMPS;

        end

        # perform right to left contraction
        colEnvironments_R = Array{Array{Any,1},1}(undef, Ly);
        for idy = Ly : -1 : 1

            # construct boundaryMPS
            boundaryMPS = Array{Any,1}(undef,3);
            if idy == Ly

                boundaryMPS[1] = rowEnvironments_U[idx][Ly + 2];
                boundaryMPS[2] = envTensors[idx,idy][4];
                boundaryMPS[3] = rowEnvironments_D[idx][Ly + 2];

            else

                @tensor boundaryMPS[1][-1 -2 -3 -4 -5; ()] := rowEnvironments_U[idx][idy + 2][-1 -2 -3 -4 1] * colEnvironments_R[idy + 1][1][1 -5];
                @tensor boundaryMPS[2][-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11] := finitePEPS[idx,idy + 1][-1 -4 4 5 -11] * finitePEPO[idx,idy + 1][-2 -5 2 3 -10 4] * conj(finitePEPS[idx,idy + 1][-3 -6 2 1 -9]) * colEnvironments_R[idy + 1][2][5 3 1 -7 -8];
                @tensor boundaryMPS[3][-1; -2 -3 -4 -5] := rowEnvironments_D[idx][idy + 2][-1 1 -3 -4 -5] * colEnvironments_R[idy + 1][3][1 -2];

                # use SVD to compress boundaryMPS
                @tensor twoSite[-1 -2 -3 -4 -5 -6 -7 -8; ()] := boundaryMPS[1][-1 1 2 3 4] * boundaryMPS[2][-2 -3 -4 -5 -6 -7 -8 4 3 2 1];
                U, S, V, ϵ = tsvd(twoSite, (1,), (2,3,4,5,6,7,8), trunc = truncdim(chiE));
                @tensor newU[-1 -2; ()] := permute(U, (1,2), ())[-1 1] * sqrt(S)[1 -2];
                @tensor newV[-1 -2 -3 -4 -5 -6 -7; -8] := sqrt(S)[-8 1] * permute(V, (2,3,4,5,6,7,8), (1,))[-1 -2 -3 -4 -5 -6 -7 1];
                boundaryMPS[1] = newU;
                boundaryMPS[2] = newV;

                @tensor twoSite[-1 -2 -3 -4; -5] := boundaryMPS[2][-1 -2 -3 1 2 3 4 -5] * boundaryMPS[3][-4 4 3 2 1];
                U, S, V, ϵ = tsvd(twoSite, (5,1,2,3), (4,), trunc = truncdim(chiE));
                @tensor newU[-1 -2 -3 -4; -5] := permute(U, (2,3,4,5), (1,))[-1 -2 -3 1 -5] * sqrt(S)[1 -4];
                @tensor newV[-1; -2]:= sqrt(S)[-2 1] * permute(V, (2,), (1,))[-1 1];
                boundaryMPS[2] = newU;
                boundaryMPS[3] = newV;

            end

            # store boundaryMPS
            colEnvironments_R[idy] = boundaryMPS;

        end

        # assign all CTM tensors
        for idy = 1 : Ly
            envTensors[idx,idy] = [ colEnvironments_L[idy][1] , rowEnvironments_U[idx][idy + 1] , colEnvironments_R[idy][1] , colEnvironments_R[idy][2] , colEnvironments_R[idy][3] , rowEnvironments_D[idx][idy + 1] , colEnvironments_L[idy][3] , colEnvironments_L[idy][2] ];
        end

    end

    # function return
    return envTensors;

end