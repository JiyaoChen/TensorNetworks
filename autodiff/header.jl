# set project directory
if ~any(occursin.(pwd(), LOAD_PATH))
    push!(LOAD_PATH, pwd())
end

# clear console
Base.run(`clear`)

# load packages
using LinearAlgebra
using OMEinsum
using Optim
using Printf
using Zygote