function computeSingleSiteExpVal(finitePEPS,envTensors)

    # contract tensor network
    expVal = @tensor envTensors[7][1 2] * envTensors[6][1 3 4 8] * envTensors[8][2 5 9 6] * finitePEPS[9 8 7 12 14] * conj(finitePEPS[5 4 7 11 13]) * envTensors[5][3 10] * envTensors[4][12 11 10 16] * envTensors[1][6 15] * envTensors[2][15 14 13 17] * envTensors[3][17 16];
    return expVal;

end