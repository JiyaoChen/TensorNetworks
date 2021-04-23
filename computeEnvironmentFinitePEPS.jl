function computeEnvironmentFinitePEPS(finitePEPS, chiE)

    # get size of finitePEPS
    (Lx, Ly) = size(finitePEPS);

    # initialize vectorSpace along the perimeter
    vecSpaceTriv = ℂ^1;

    # initialize CTM tensors around the finitePEPS circumference
    envTensors = Array{Array{Any,1},2}(undef, Lx, Ly);
    for idx = 1 : Lx
        for idy = 1 : Ly
            
            # get PEPS tensor
            bulkTensor = finitePEPS[idx, idy];

            # initialize CTM tensors
            C1 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv);
            T1 = TensorMap(ones, vecSpaceTriv ⊗ space(bulkTensor,5)' ⊗ space(bulkTensor,5), vecSpaceTriv);
            C2 = TensorMap(ones, vecSpaceTriv ⊗ vecSpaceTriv, one(vecSpaceTriv));
            T2 = TensorMap(ones, space(bulkTensor,4)' ⊗ space(bulkTensor,4) ⊗ vecSpaceTriv, vecSpaceTriv);
            C3 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv);
            T3 = TensorMap(ones, vecSpaceTriv, vecSpaceTriv ⊗ space(bulkTensor,2)' ⊗ space(bulkTensor,2));
            C4 = TensorMap(ones, one(vecSpaceTriv), vecSpaceTriv ⊗ vecSpaceTriv);
            T4 = TensorMap(ones, vecSpaceTriv, space(bulkTensor,1)' ⊗ space(bulkTensor,1) ⊗ vecSpaceTriv);
            
            # store CTM tensors
            envTensors[idx,idy] = [C1, T1, C2, T2, C3, T3, C4, T4];

        end
    end


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

            @tensor boundaryMPS[1][-1; -2 -3 -4] := envTensors[idx - 1,1][8][-1 -2 -3 1] * rowEnvironments_U[idx - 1][1][1 -4];
            for idy = 1 : Ly
                @tensor boundaryMPS[idy + 1][-1 -2 -3 -4 -5; -6 -7 -8] := finitePEPS[idx - 1,idy][-2 -4 2 -7 3] * conj(finitePEPS[idx - 1,idy][-3 -5 2 -6 1]) * rowEnvironments_U[idx - 1][idy + 1][-1 3 1 -8];
            end
            @tensor boundaryMPS[Ly + 2][-1 -2 -3 -4; ()] := envTensors[idx - 1,Ly][4][-2 -3 -4 1] * rowEnvironments_U[idx - 1][Ly + 2][-1 1];
            
            # use SVD to compress boundaryMPS
            for idy = 1 : Ly + 1

                if idy == 1
                    @tensor twoSite[-1 -2 -3; -4 -5 -6] := boundaryMPS[idy][-1 1 2 3] * boundaryMPS[idy + 1][3 2 1 -2 -3 -4 -5 -6];
                    U, S, V, ϵ = tsvd(twoSite, (1,), (2,3,4,5,6), trunc = truncdim(chiE));
                    @tensor newU[-1; -2] := U[-1 1] * sqrt(S)[1 -2];
                    @tensor newV[-1 -2 -3; -4 -5 -6] := sqrt(S)[-1 1] * permute(V,(1,2,3), (4,5,6))[1 -2 -3 -4 -5 -6];
                elseif idy == Ly + 1
                    @tensor twoSite[-1 -2 -3 -4; ()] := boundaryMPS[idy][-1 -2 -3 1 2 3] * boundaryMPS[idy + 1][3 2 1 -4];
                    U, S, V, ϵ = tsvd(twoSite, (1,2,3), (4,), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3; -4] := U[-1 -2 -3 1] * sqrt(S)[1 -4];
                    @tensor newV[-1 -2; ()] := sqrt(S)[-1 1] * permute(V,(1,2), ())[1 -2];
                else
                    @tensor twoSite[-1 -2 -3 -4 -5; -6 -7 -8] := boundaryMPS[idy][-1 -2 -3 1 2 3] * boundaryMPS[idy + 1][3 2 1 -4 -5 -6 -7 -8];
                    U, S, V, ϵ = tsvd(twoSite, (1,2,3), (4,5,6,7,8), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3; -4] := U[-1 -2 -3 1] * sqrt(S)[1 -4];
                    @tensor newV[-1 -2 -3; -4 -5 -6] := sqrt(S)[-1 1] * permute(V,(1,2,3), (4,5,6))[1 -2 -3 -4 -5 -6];
                end

                boundaryMPS[idy] = newU;
                boundaryMPS[idy + 1] = newV;

            end

        end

        # store boundaryMPS
        rowEnvironments_U[idx] = boundaryMPS;

    end

    # compute effective environments using boundary MPS methods from below
    rowEnvironments_D = Array{Array{Any,1},1}(undef,Lx);
    for idx = Lx : -1 : 1

        # construct boundaryMPS
        boundaryMPS = Array{Any,1}(undef,Ly + 2);
        if idx == Lx

            boundaryMPS[1] = envTensors[idx,1][7];
            for idy = 1 : Ly
                boundaryMPS[idy + 1] = envTensors[idx,idy][6];
            end
            boundaryMPS[Ly + 2] = envTensors[idx,Ly][5];

        else

            @tensor boundaryMPS[1][(); -1 -2 -3 -4] := rowEnvironments_D[idx + 1][1][1 -1] * envTensors[idx + 1,1][8][1 -2 -3 -4];
            for idy = 1 : Ly
                @tensor boundaryMPS[idy + 1][-1 -2 -3; -4 -5 -6 -7 -8] := rowEnvironments_D[idx + 1][idy + 1][-3 -4 1 3] * finitePEPS[idx + 1,idy][-1 3 2 -6 -8] * conj(finitePEPS[idx + 1,idy][-2 1 2 -5 -7]);
            end
            @tensor boundaryMPS[Ly + 2][-1 -2 -3; -4] := rowEnvironments_D[idx + 1][Ly + 2][-3 1] * envTensors[idx + 1,Ly][4][-1 -2 1 -4];

            # use SVD to compress boundaryMPS
            for idy = Ly + 2 : -1 : 2

                if idy == Ly + 2
                    @tensor twoSite[-1 -2 -3; -4 -5 -6] := boundaryMPS[idy - 1][-1 -2 -3 1 2 3 -5 -6] * boundaryMPS[idy][3 2 1 -4];
                    U, S, V, ϵ = tsvd(twoSite, (5,6,1,2,3), (4,), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3; -4 -5 -6] := permute(U, (3,4,5), (6,1,2))[-1 -2 -3 1 -5 -6] * sqrt(S)[1 -4];
                    @tensor newV[-1; -2] := sqrt(S)[-1 1] * V[1 -2];
                elseif idy == 2
                    @tensor twoSite[(); -1 -2 -3 -4] := boundaryMPS[idy - 1][1 2 3 -4] * boundaryMPS[idy][3 2 1 -1 -2 -3];
                    U, S, V, ϵ = tsvd(twoSite, (4,), (1,2,3), trunc = truncdim(chiE));
                    @tensor newU[(); -1 -2] := permute(U, (), (2,1))[1 -2] * sqrt(S)[1 -1];
                    @tensor newV[-1; -2 -3 -4] := sqrt(S)[-1 1] * V[1 -2 -3 -4];
                else
                    @tensor twoSite[-1 -2 -3; -4 -5 -6 -7 -8] := boundaryMPS[idy - 1][-1 -2 -3 1 2 3 -7 -8] * boundaryMPS[idy][3 2 1 -4 -5 -6];
                    U, S, V, ϵ = tsvd(twoSite, (7,8,1,2,3), (4,5,6), trunc = truncdim(chiE));
                    @tensor newU[-1 -2 -3; -4 -5 -6] := permute(U, (3,4,5), (6,1,2))[-1 -2 -3 1 -5 -6] * sqrt(S)[1 -4];
                    @tensor newV[-1; -2 -3 -4] := sqrt(S)[-1 1] * V[1 -2 -3 -4];
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
        colEnvironments_L = Array{Array{Any,1},1}(undef,Ly);
        for idy = 1 : Ly

            # construct boundaryMPS
            boundaryMPS = Array{Any,1}(undef,3);
            if idy == 1

                boundaryMPS[1] = rowEnvironments_U[idx][1];
                boundaryMPS[2] = envTensors[idx,idy][8];
                boundaryMPS[3] = rowEnvironments_D[idx][1];

            else

                @tensor boundaryMPS[1][-1 -2 -3; -4] := colEnvironments_L[idy - 1][1][-1 1] * rowEnvironments_U[idx][idy][1 -2 -3 -4];
                @tensor boundaryMPS[2][-1 -2 -3; -4 -5 -6 -7 -8] := colEnvironments_L[idy - 1][2][-1 1 3 -8] * finitePEPS[idx,idy - 1][3 -2 2 -5 -7] * conj(finitePEPS[idx,idy - 1][1 -3 2 -4 -6]);
                @tensor boundaryMPS[3][(); -1 -2 -3 -4] := colEnvironments_L[idy - 1][3][1 -4] * rowEnvironments_D[idx][idy][1 -1 -2 -3];

                # use SVD to compress boundaryMPS
                @tensor twoSite[-1 -2 -3; -4 -5 -6] := boundaryMPS[2][-1 -2 -3 -4 -5 1 2 3] * boundaryMPS[1][3 2 1 -6];
                U, S, V, ϵ = tsvd(twoSite, (1,2,3,4,5), (6,), trunc = truncdim(chiE));
                @tensor newU[-1 -2 -3; -4 -5 -6] := permute(U, (1,2,3), (4,5,6))[-1 -2 -3 -4 -5 1] * sqrt(S)[1 -6];
                @tensor newV[-1; -2] := sqrt(S)[-1 1] * V[1 -2];
                boundaryMPS[1] = newV;
                boundaryMPS[2] = newU;

                @tensor twoSite[(); -1 -2 -3 -4] := boundaryMPS[3][-1 1 2 3] * boundaryMPS[2][3 2 1 -2 -3 -4];
                U, S, V, ϵ = tsvd(twoSite, (1,), (2,3,4), trunc = truncdim(chiE));
                @tensor newU[(); -1 -2] := permute(U, (), (1,2))[-1 1] * sqrt(S)[1 -2];
                @tensor newV[-1; -2 -3 -4] := sqrt(S)[-1 1] * V[1 -2 -3 -4];
                boundaryMPS[2] = newV;
                boundaryMPS[3] = newU;

            end

            # store boundaryMPS
            colEnvironments_L[idy] = boundaryMPS;

        end

        # perform right to left contraction
        colEnvironments_R = Array{Array{Any,1},1}(undef,Ly);
        for idy = Ly : -1 : 1

            # construct boundaryMPS
            boundaryMPS = Array{Any,1}(undef,3);
            if idy == Ly

                boundaryMPS[1] = rowEnvironments_U[idx][Ly + 2];
                boundaryMPS[2] = envTensors[idx,idy][4];
                boundaryMPS[3] = rowEnvironments_D[idx][Ly + 2];

            else

                @tensor boundaryMPS[1][-1 -2 -3 -4; ()] := rowEnvironments_U[idx][idy + 2][-1 -2 -3 1] * colEnvironments_R[idy + 1][1][1 -4];
                @tensor boundaryMPS[2][-1 -2 -3 -4 -5; -6 -7 -8] := finitePEPS[idx,idy + 1][-1 -3 2 3 -8] * conj(finitePEPS[idx,idy + 1][-2 -4 2 1 -7]) * colEnvironments_R[idy + 1][2][3 1 -5 -6];
                @tensor boundaryMPS[3][-1; -2 -3 -4] := rowEnvironments_D[idx][idy + 2][-1 1 -3 -4] * colEnvironments_R[idy + 1][3][1 -2];

                # use SVD to compress boundaryMPS
                @tensor twoSite[-1 -2 -3 -4 -5 -6; ()] := boundaryMPS[1][-1 1 2 3] * boundaryMPS[2][-2 -3 -4 -5 -6 3 2 1];
                U, S, V, ϵ = tsvd(twoSite, (1,), (2,3,4,5,6), trunc = truncdim(chiE));
                @tensor newU[-1 -2; ()] := permute(U, (1,2), ())[-1 1] * sqrt(S)[1 -2];
                @tensor newV[-1 -2 -3 -4 -5; -6] := sqrt(S)[-6 1] * permute(V, (2,3,4,5,6), (1,))[-1 -2 -3 -4 -5 1];
                boundaryMPS[1] = newU;
                boundaryMPS[2] = newV;

                @tensor twoSite[-1 -2 -3; -4] := boundaryMPS[2][-1 -2 3 2 1 -4] * boundaryMPS[3][-3 1 2 3];
                U, S, V, ϵ = tsvd(twoSite, (4,1,2), (3,), trunc = truncdim(chiE));
                @tensor newU[-1 -2 -3; -4] := permute(U, (2,3,4), (1,))[-1 -2 1 -4] * sqrt(S)[1 -3];
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