function relaxtion1(node::SplitNode)
    l̅, s₁ = node.bounds[1, 1], node.bounds[1, 2]
    s₂, u̅ = node.bounds[2, 1], node.bounds[2, 2]
    
    λ = u̅ / (u̅ - l̅)
    γ = 0.5 * λ * (s₁ - l̅)
    ĉ = λ * (Z.c - l̅) - γ
end