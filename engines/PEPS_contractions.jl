function applyNORM_OneSite(finitePEPS, envTensors)

    # contract tensor network
    @tensor NPsi[-1 -2 -3 -4 -5] := envTensors[1][6 12] * envTensors[2][12 11 -5 10] * envTensors[3][10 9] * envTensors[4][8 -4 7 9] *
        envTensors[5][3 7] * envTensors[6][1 3 -2 4] * envTensors[7][1 2] * envTensors[8][2 -1 5 6] *
        finitePEPS[5 4 -3 8 11];

    return NPsi;
end

function applyPEPO_OneSite(finitePEPS, finitePEPO, envTensors)

    # contract tensor network
    @tensor HPsi[-1 -2 -3 -4 -5] := envTensors[1][17 4] * envTensors[2][4 11 8 -5 2] * envTensors[3][2 1] * envTensors[4][10 7 -4 3 1] *
    envTensors[5][13 3] * envTensors[6][21 13 -2 16 15] * envTensors[7][21 22] * envTensors[8][22 -1 20 19 17] *
        finitePEPS[19 15 12 10 11] * 
        finitePEPO[20 16 -3 7 8 12];

    return HPsi;
end