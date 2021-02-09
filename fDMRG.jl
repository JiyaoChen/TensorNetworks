using KrylovKit
using TensorKit

# parameters Ising model
J = 3.0;
h = 9.0;
maxNumSteps = convert(Int64,1e1);
χ = 100;

# Pauli opertators
Id = [+1 0; 0 +1];
X = [0 +1; +1 0];
Y = [0 -1im; +1im 0];
Z = [+1 0; 0 -1];

# generate the Ising MPO
# legs
link_phys = ℤ₂Space(0 => 1, 1 => 1);
mpo_left_link = ℤ₂Space(0 => 2, 1 => 1);
mpo_right_link = ℤ₂Space(0 => 2, 1 => 1);
# operators
ham_arr = zeros(Float64, dim(mpo_left_link), dim(link_phys), dim(mpo_right_link), dim(link_phys));
ham_arr[1,:,1,:] = Id;
ham_arr[2,:,2,:] = Id;
ham_arr[1,:,3,:] = J*X;
ham_arr[3,:,2,:] = X;
ham_arr[1,:,2,:] = h*Z;
# write the array to the mpo
ham_mpo = Tensor(ham_arr, mpo_left_link*link_phys*mpo_right_link*link_phys);

# generate the first MPS tensors
# legs
mps_left_link = ℤ₂Space(0=>1);
mps_right_link = ℤ₂Space(0=>1);
mps_shared_link = fuse(mps_left_link, link_phys);
mps_tensor_1 = Tensor(ones, mps_left_link*link_phys*mps_shared_link);
mps_tensor_2 = Tensor(ones, mps_shared_link*link_phys*mps_right_link);

fusiontrees(ham_mpo)
