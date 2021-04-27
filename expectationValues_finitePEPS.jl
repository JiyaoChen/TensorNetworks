function computeSingleSiteExpVal(finitePEPS, envTensors)

    # contract tensor network
    expVal = @tensor envTensors[1][6 15] * envTensors[2][15 14 13 17] * envTensors[3][17 16] * envTensors[4][12 11 10 16] *
        envTensors[5][3 10] * envTensors[6][1 3 4 7] * envTensors[7][1 2] * envTensors[8][2 5 8 6] *
        finitePEPS[8 7 9 12 14] * 
        conj(finitePEPS[5 4 9 11 13]);

    return expVal;
end

function computeSingleSiteExpVal_PEPO(tensorPEPS, tensorPEPO, envTensors)

    # contract tensor network
    expVal = @tensor envTensors[1][17 4] * envTensors[2][4 11 8 6 2] * envTensors[3][2 1] * envTensors[4][10 7 5 3 1] *
    envTensors[5][13 3] * envTensors[6][21 13 14 16 15] * envTensors[7][21 22] * envTensors[8][22 18 20 19 17] *
        tensorPEPS[19 15 12 10 11] * 
        tensorPEPO[20 16 9 7 8 12] *
        conj(tensorPEPS[18 14 9 5 6]);

    return expVal;
end