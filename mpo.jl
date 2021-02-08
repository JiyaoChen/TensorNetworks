using TensorKit

# vL = ℤ₂Space(0 => 2, 1=> 2)
# vP = ℤ₂Space(0 => 1, 1=> 1)
# sz_arr = [1 0; 0 -1]
# sx_arr = [0 1; 1  0]
# sz_map = TensorMap(sz_arr, vP, vP)
# sz_map_u0 = convert(Array, sz_arr)
# print("\n", sz_map_u0, "\n")

# tensorG = Tensor(ones, vL*vP*vL)
# print(@tensor outTensor[a, d, c] := sz_map[d,b]*tensorG[a,b,c])

# # U1 example
# vPhys = U1Space(1 => 1)
# vVirt = U1Space(0 => 3,1 => 2)
# tensorG = TensorMap(randn, vPhys*vVirt, vVirt)
# tensorL = TensorMap(randn, vVirt, vVirt)
# print("\n",tensorL,"\n")
# print("\n",convert(Array,tensorL),"\n")

# SU2 example
vP = SU2Space(1/2 => 1);
vV = SU2Space(0 => 3,1/2 => 2);

tensorG = TensorMap(randn, vP * vV,vV);
# print("\n",tensorG,"\n")
# print("\n",convert(Array,tensorG),"\n")

tensorL = TensorMap(randn, vV, vV);
D, V = eigen(tensorL);
print(V)
# print("\n",tensorL,"\n")
# print("\n",convert(Array,tensorL),"\n")

# tensorM = TensorMap(zeros, vP * vV, vV * vP);

# @tensor tensorGL[-1 -2 -3] := tensorG[-1 -2 1] * tensorL[1 -3]
# print(tensorGL)

# rand_t = TensorMap(randn, v*v, v)

# @tensor new_tensor[a, c] := sz_map[a, b]*sz_map[b, c]

# sz_arr = [1 0; 0 -1]
# Tensor(sz_arr, v^2)
# convert(sz_arr, Tensor)

# for s in sectors(mpo)
#     @show s, dim(mpo, s)
# end