using Plots, Random, LinearAlgebra, LightGraphs, GraphPlot, Combinatorics, StatsBase
const ⊗ = kron

#Define Hadamard gate and |+>=H|0> state
H = [1 1; 1 -1]/sqrt(2);
qubit_plus = H*[1,0];

"""
Creates a 1D graph chain 
"""
function graph1D(n::Int)
    G = SimpleGraph(n)
    for i in 1:n-1
        add_edge!(G, i, i+1);
    end
    G
end


"""
Matrix representation of operator CZij in n qubits

Input: i,j,n
Output: CZij
"""
function controlled_z(i::Int,j::Int,n::Int)
    op1, op2 = 1,2
    for k in 1:n
        op1 = op1 ⊗ I(2)
        op2 = k==i || k==j ? op2 ⊗ ([0,1]'⊗[0,1]) : op2 ⊗ I(2)
    end
    op1-op2
end


"""
Matrix representation of operator SWAPij in n qubits

Input: i,j,n
Output: SWAPij
"""
function swap_ij(i::Int , j::Int, n::Int)
    op1, op2, op3, op4 = ones(4)
    for k in 1:n
        op1 = k==i || k==j ? op1 ⊗ ([1,0]'⊗[1,0]) : op1 ⊗ I(2)
        op4 = k==i || k==j ? op4 ⊗ ([0,1]'⊗[0,1]) : op4 ⊗ I(2)
        if k==i
            op2, op3 = op2 ⊗ ([1,0]'⊗[0,1]), op3 ⊗ ([0,1]'⊗[1,0]) 
        elseif k==j
            op2, op3 = op2 ⊗ ([0,1]'⊗[1,0]), op3 ⊗ ([1,0]'⊗[0,1]) 
        else
            op2, op3 = op2 ⊗ I(2), op3 ⊗ I(2)
        end
    end
    op1+op2+op3+op4
end


"""  
Input:
G = graph

Output:
ψ = Π_{(i,j)∈ G}(CZij)(|+>)^⊗n  
"""
function create_graph_state(G)
    ψ = 1
    for i in 1:nv(G)
        ψ = ψ⊗qubit_plus
    end
    for e in edges(G)
        ψ=controlled_z(src(e),dst(e),nv(G))*ψ
    end
    ψ
end


"""
Same as create_graph_state but qubit 1 is ψ_in
(only works with ψ_in pure and single qubit)
"""
function graph_with_input(ψ_in, G)
    for i in 2:nv(G)
        ψ_in = ψ_in⊗qubit_plus
    end
    for e in edges(G)
        ψ_in=controlled_z(src(e),dst(e),nv(G))*ψ_in
    end
    ψ_in
end

"""Pure state to density matrix
"""
pure2density(ψ::Vector) = ψ ⊗ ψ'

"""Fidelity between ρ and σ
"""
function fidelity(ρ, σ)
    sρ = sqrt(ρ)
    return abs2(tr(sqrt(sρ*σ*sρ)))
end


"""
Measures ith qubit of ρ in z basis and updates the graph edges
"""
function measure_z(graph, ρ, i::Int; fix=true)
    n = nv(graph)
    nbh = neighbors(graph, i)

    ρ = ρ isa Vector ? pure2density(ρ) : ρ
    pi_0, pi_1 = 1, 1
    for k in 1:n
        pi_0, pi_1 = k==i ? (pi_0 ⊗ [1 0; 0 0]  , pi_1 ⊗ [0 0; 0 1]) : (pi_0 ⊗ I(2) , pi_1 ⊗ I(2))
    end
    prob0, prob1 =real(tr(ρ*pi_0)), real(tr(ρ*pi_1))
    measurement = sample([0,1], pweights([prob0,prob1]))
    ρ = measurement==0 ? pi_0*ρ*pi_0/prob0 : pi_1*ρ*pi_1/prob1
    
    #fix ρ
    if measurement==1 && fix
        u_fix = 1
        for k in 1:n
            u_fix = k in nbh ? u_fix ⊗ [1 0; 0 -1] : u_fix ⊗ I(2)
        end
        ρ = u_fix*ρ*(u_fix')
    end
    return ρ, measurement 
end


"""
Measures ith qubit in basis {|0>+exp(iφ)|1>, |0>-exp(iφ)|1>}/sqrt(2)

Input: 
ρ     = vector representation of ψ
ϕ     = angle of measurement basis
i     = measured qubit

Output:
ρ           = new state after measurement
measurement = 0 or 1
"""
function measure_angle(ρ, ϕ, i::Int)
    ρ = ρ isa Vector ? pure2density(ρ) : ρ
    n = floor(Int,log(2,size(ρ, 1)))
    pi_0, pi_1 = 1, 1
    for k in 1:n
        pi_0, pi_1 = k==i ? (pi_0 ⊗ [1 exp(-im*ϕ); exp(im*ϕ) 1]/2  , pi_1 ⊗ [1 -exp(-im*ϕ);-exp(im*ϕ) 1]/2 ) : (pi_0 ⊗ I(2) , pi_1 ⊗ I(2))
    end
    prob0, prob1 = real(tr(ρ*pi_0)), real(tr(ρ*pi_1))
    measurement = sample([0,1], pweights([prob0,prob1]))
    
    ρ = measurement==0 ? pi_0*ρ*pi_0/prob0 : pi_1*ρ*pi_1/prob1   
    return ρ, measurement
end


"""
Input:
ρ       = state
indices = to be traced

Output
σ  = Tr_{indices} ρ
"""
function partial_trace(ρ, indices)
    x,y = size(ρ)
    n = floor(Int,log(2,x))
    r = size(indices,1)
    σ = zeros(floor(Int, x/(2^r)), floor(Int,y/(2^r)))
    for m in 1:2^r
        qubits = digits(m, base=2, pad=r)
        ptrace = 1
        for k in 1:n
            if k in indices
                ptrace = qubits[findfirst(x->x==k, indices)]==0 ? ptrace ⊗ [1,0] : ptrace ⊗ [0,1] 
            else
                ptrace = ptrace ⊗ I(2)
            end
        end
        σ += (ptrace')*ρ*ptrace
    end
    return σ
end


"""
Random unitary in N dimensions
taken from https://discourse.julialang.org/t/how-to-generate-a-random-unitary-matrix-perfectly-in-julia/34102
"""
function RandomUnitaryMatrix(N::Int)
    x = (rand(N,N) + rand(N,N)*im) / sqrt(2)
    f = qr(x)
    diagR = sign.(real(diag(f.R)))
    diagR[diagR.==0] .= 1
    diagRm = diagm(diagR)
    u = f.Q * diagRm
    return u
end


"""
Input: 
ρ  = vector/matrix representing quantum state
sx = vectors of 0s and 1s
sy = vectors of 0s and 1s

Output: ((σ_x)^⊗sx (σ_z)^⊗sz)ρ((σ_x)^⊗sx (σ_z)^⊗sz)
"""
function apply_byproduct(ρ, sx, sz)
    ρ = ρ isa Vector ? pure2density(ρ) : ρ
    n = floor(Int,log(2,size(ρ, 1)))
    byprod = 1
    for i in 1:n
        byprod = byprod ⊗ ( ([0 1; 1 0]^(sx[i])) * [1 0; 0 -1]^(sz[i]) )
    end
    return byprod*ρ*byprod'
end


"""
info : measurement outcomes
state: angles
Tracks byproduct operator
op = 0 -> Z
op = 1 -> X
op = 2 -> Y

returns: sx and sy string
"""
function byproduct_track(info, state)
    sx = 0
    sz = 0
    for (ind_i,i) in enumerate(info)
        if i==0
            continue
        elseif i==1
            op = mod(ind_i,2)==0 ? 1 : 0
            for (ind_j,j) in enumerate(state)
                if ind_j>ind_i
                    ac = mod(ind_j,2)==0 ? 1 : 0
                    if j==1
                        continue
                    elseif j==-1
                        if ac!=op
                            op = setdiff!([0,1,2], [op,ac])[1]
                        end
                    end
                end
            end
            if op==0
                sx,sz = sx, mod(sz+1,2)
            elseif op==1
                sx,sz = mod(sx+1,2), sz
            elseif op==2
                sx,sz = mod(sx+1,2), mod(sz+1,2)
            end
        end
    end
    sx, sz
end



"""
Returns pure graph state with pure multiple inputs
"""
function graph_with_multiple_inputs(G ; inputs=[], indices=[])
    @assert size(inputs)==size(indices)
    ψ = 1
    if isempty(inputs)
        ψ = create_graph_state(G) 
    elseif !isempty(inputs)
        for i in 1:nv(G)
            st = i in indices ? inputs[findfirst(x->x==i, indices)] : qubit_plus
            ψ = ψ ⊗ st
        end
        for e in edges(G)
            ψ=controlled_z(src(e),dst(e),nv(G))*ψ
        end 
    end
    ψ
end

"""
Applies cz_{i,i+1} to every qubit
"""
function cz_after_layer_measurement(ρ, n::Int)
    ρ = ρ isa Vector ? pure2density(ρ) : ρ
    if n>1
        for i in 1:n-1
            controll = controlled_z(i, i+1, n)
            ρ = controll*ρ*(controll')
        end
    end
    ρ
end

"""
Measures angle in 2d state
"""
function measure_angle_2d_intermediate(ρ, ϕ, i::Int, n::Int)
    #entangle extra state with ith qubit
    ρ = ρ ⊗ pure2density(qubit_plus)
    controll_ilast = controlled_z(i, n+1, n+1)
    ρ = controll_ilast*ρ*(controll_ilast')
    
    #measures
    ρ, outcome = measure_angle(ρ, ϕ, i)
    swap_ilast = swap_ij(i, n+1,n+1)
    ρ = swap_ilast*ρ*(swap_ilast')
    ρ = partial_trace(ρ, [n+1])
    return ρ, outcome
end

"""
Measures z in 2d state
"""
function measure_z_2d_intermediate(ρ, i::Int, n::Int)
    graph = graph1D(n)
    add_vertex!(graph)
    add_edge!(graph, i, n+1)
    
    #entangle extra state with ith qubit
    ρ = ρ ⊗ pure2density(qubit_plus)
    controll_ilast = controlled_z(i, n+1, n+1)
    ρ = controll_ilast*ρ*(controll_ilast')
    
    #measures
    ρ, outcome = measure_z(graph, ρ, i::Int; fix=true)
    swap_ilast = swap_ij(i, n+1,n+1)
    ρ = swap_ilast*ρ*(swap_ilast')
    ρ = partial_trace(ρ, [n+1])
    return ρ, outcome
end


"""
Measures ith qubit of ρ with an angle ϕ∈[0,2π]. If ϕ=-1, it measures in Z basis.
"""
function layer_measurement(ρ, ϕ, i::Int, n::Int, last_layer)
    if !last_layer
        ρ, outcome = ϕ ==-1 ? measure_z_2d_intermediate(ρ, i, n) : measure_angle_2d_intermediate(ρ, ϕ, i, n)
    elseif last_layer
        ρ, outcome = ϕ==-1 ? measure_z(graph1D(n), ρ, i::Int; fix=true) : measure_angle(ρ, ϕ, i)
    end
    ρ, outcome
end


"""
Matrix representation of operator CNOTij in n qubits

Input: i,j,n
Output: CNOTij
"""
function cnot_ij(i::Int , j::Int, n::Int)
    op1, op2, op3, op4 = ones(4)
    for k in 1:n
        op1 = k==i || k==j ? op1 ⊗ ([1,0]'⊗[1,0]) : op1 ⊗ I(2)
        if k==i
            op2, op3, op4 =op2 ⊗ ([1,0]'⊗[1,0]) , op3 ⊗ ([0,1]'⊗[0,1]), op4 ⊗ ([0,1]'⊗[0,1]) 
        elseif k==j
            op2, op3, op4 =op2 ⊗ ([0,1]'⊗[0,1]) ,op3 ⊗ ([1,0]'⊗[0,1]), op4 ⊗ ([0,1]'⊗[1,0]) 
        else
            op2, op3, op4 = op2 ⊗ I(2), op3 ⊗ I(2), op4 ⊗ I(2)
        end
    end
    op1+op2+op3+op4
end



