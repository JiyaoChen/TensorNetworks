σ₀ = [1.0 0.0; 0.0 1.0]
σ₁ = [0.0 1.0; 1.0 0.0]
σ₃ = [1.0 0.0; 0.0 -1.0]
σ₂ = -1im*σ₃*σ₁

function isingTBG(h, J; id = 0.0)
    reshape(h*(kron(σ₃, σ₀) + kron(σ₀, σ₃)) + J*kron(σ₁, σ₁) + id*kron(σ₀, σ₀), (2,2,2,2))
end

function heisenbergTBG(Jx, Jy, Jz, h; id = 0.0)
    return 0.5*real(reshape(Jx*kron(σ₁, σ₁) + Jy*kron(σ₂, σ₂) + Jz*kron(σ₃, σ₃) + h*(kron(σ₃, σ₀) + kron(σ₀, σ₃)) + id*kron(σ₀, σ₀), (2, 2, 2, 2)))
end