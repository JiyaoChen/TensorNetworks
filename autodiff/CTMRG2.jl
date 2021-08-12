# include functions
include("performAbsorption_CTM.jl")
include("rotateLatticePEPS_A90.jl")

# CTMRG procedure for arbitrary unit cell
function runCTMRG(gammaTensors, environmentTensors, unitCellCTM, chiE, truncBelowE, convTol_Env; verbosePrint = false)

    # get system size
    Lx, Ly = size(gammaTensors);

    # maximal number of CTMRG steps
    maxNumSteps = Integer(1e2);

    # initialize cell of singular values of corner tensors
    singularValuesCornerTensors = Array{Array{Float64,3},2}(undef, Lx, Ly);
    for x = 1 : Lx
        for y = 1 : Ly
            singularValues = zeros(Float64, maxNumSteps, chiE, 4);
            singularValuesCornerTensors[x,y] = singularValues;
        end
    end

    # parameters to control the while loop
    runDirectedCTM = 1;

    # print CTMRG info
    # @printf("\nRunning CTMRG...\n")

    # perform directional update
    envLoopCounter = 1;
    normSingularValues = 1;
    while runDirectedCTM == 1 && envLoopCounter <= maxNumSteps

        # perform absorption step in the left direction
        environmentTensors = performAbsorption_CTM(gammaTensors, unitCellCTM, environmentTensors, chiE, truncBelowE);        
        gammaTensors, unitCellCTM, environmentTensors = rotateLatticePEPS_A90(gammaTensors, unitCellCTM, environmentTensors);
        # println("absorption 1 done\n")

        # perform absorption step to the top
        environmentTensors = performAbsorption_CTM(gammaTensors, unitCellCTM, environmentTensors, chiE, truncBelowE);
        gammaTensors, unitCellCTM, environmentTensors = rotateLatticePEPS_A90(gammaTensors, unitCellCTM, environmentTensors);
        # println("absorption 2 done\n")

        # perform absorption step to the right
        environmentTensors = performAbsorption_CTM(gammaTensors, unitCellCTM, environmentTensors, chiE, truncBelowE);
        gammaTensors, unitCellCTM, environmentTensors = rotateLatticePEPS_A90(gammaTensors, unitCellCTM, environmentTensors);
        # println("absorption 3 done\n")

        # perform absorption step to the bottom
        environmentTensors = performAbsorption_CTM(gammaTensors, unitCellCTM, environmentTensors, chiE, truncBelowE);
        gammaTensors, unitCellCTM, environmentTensors = rotateLatticePEPS_A90(gammaTensors, unitCellCTM, environmentTensors);
        # println("absorption 4 done\n")

        # get singular values
        for x = 1 : Lx
            for y = 1 : Ly
                singularValues = singularValuesCornerTensors[x,y];
                indEnvironment = environmentTensors[x,y];
                cornerIdxs = [1 3 5 7];
                for c = 1 : length(cornerIdxs)
                    F = tsvd(indEnvironment[cornerIdxs[c]], (1,), (2,));
                    singV = getSingularValues(F[2]);
                    singularValues[envLoopCounter,1 : length(singV),c] = singV;
                end
                singularValuesCornerTensors[x,y] = singularValues;
            end
        end

        # check convergence of singular values
        if envLoopCounter > 1

            # determine norm of the difference of singular values
            normSingularValues = zeros(Float64, Lx * Ly);
            for x = 1 : Lx
                for y = 1 : Ly
                    listOfSingularValues = singularValuesCornerTensors[x,y];
                    normSingularValues[(x - 1) * Ly + y] = norm(reshape(listOfSingularValues[envLoopCounter - 0,:,:], chiE, 4) - reshape(listOfSingularValues[envLoopCounter - 1,:,:], chiE, 4));
                end
            end

            # check convergence of the environment
            if maximum(normSingularValues) < convTol_Env
                runDirectedCTM = 0;
            end

        end

        # print CTM information
        verbosePrint && @printf("CTMRG Step %03d - normSingularValues %0.8e\n", envLoopCounter, maximum(normSingularValues))

        # increase loop counter
        envLoopCounter += 1;

    end

    # return number of CTM steps necessary to achieve convergence
    numE = envLoopCounter - 1;

    # print CTM convergence info
    @printf("CTMRG with %d numSteps, normSingularValues %0.8e\n", numE, maximum(normSingularValues))

    # function return
    return environmentTensors, numE;

end
