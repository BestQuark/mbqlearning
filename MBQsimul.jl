using Plots, Random, LinearAlgebra, LightGraphs, GraphPlot, Combinatorics, StatsBase
const ⊗ = kron

#Define Hadamard gate and |+>=H|0> state
H = [1 1; 1 -1]/sqrt(2);
qubit_plus = H*[1,0];
pz = [1 0; 0 im];
X = [0 1; 1 0];
Z = [1 0; 0 -1];
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

function arbitrary_qubit_gate(u::Matrix, i::Int, n::Int)
    """
    Matrix representation of operator U in n qubits
    
    Input: U,i,n
    Output: U
    """
    op = 1
    for k in 1:n
        op = k==i ? op ⊗ u : op ⊗ I(2)
    end
    op
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


##------------------1D RL functions:-----------------

function reward(ρ, σ)
    """
    calculates the reward between two states ρ, σ defined by
    max_{P pauli} F(P*ρ*P, σ)
    """
    rw = 0
    n = floor(Int,log(2,size(ρ, 1)))
    paulix, pauliz = [0 1; 1 0], [1 0; 0 -1]
    for m1 in 1:2^n, m2 in 1:2^n
        str_x = digits(m1, base=2, pad=n)
        str_z = digits(m2, base=2, pad=n)
        opx,opz = 1,1
        for i in 1:n, j in 1:n
            opx = str_x[i]==1 ? opx ⊗ paulix : opx ⊗ I(2)
            opz = str_z[i]==1 ? opz ⊗ pauliz : opz ⊗ I(2)
            rw = max(rw, fidelity(opx*opz*ρ*opz*opx,σ))
        end
    end
    return rw
end

function prepare_env_states(n)
    """
    Prepares observation space:
    Every element is of the form
    [±1,.., ±1, 0, ...,0]
    Where the first i elements are ±1 depending on the outcome of the measurements ϕ_j j∈{1,...,i} and 0 elsewhere
    """
    outcomes = [-1,1]
    states = [zeros(n)]
    for i in 1:n-1
        ll = []
        for st in states[2^(i-1): end]
            cst = copy(st)
            l = []
            for o in outcomes
                new_st = copy(cst)
                new_st[i] = o
                l = isempty(l) ? [new_st] : append!(l, [new_st])
            end
            ll = isempty(ll) ? l : hcat(ll, l)
        end
        states = append!(states, ll)
    end
    states
end


function prepare_env_states2(n)
    """
    Prepares observation space:
    Every element is of the form
    [±1,.., ±1, 0, ...,0 , a_1, ..., a_i, 0, ...,0]
    Where the first i elements are ±1 depending on the outcome of the measurements ϕ_j j∈{1,...,i} and 0 elsewhere
    """
    outcomes = [-1,1]
    actions  = [-1, 1]
    states = [zeros(2*n)]
    for i in 1:n-1
        ll = []
        for st in states[2^(i-1): end]
            cst = copy(st)
            l = []
            for o in outcomes
                for a in actions
                    new_st = copy(cst)
                    new_st[i] = o
                    new_st[n+i] = a
                    l = isempty(l) ? [new_st] : append!(l, [new_st])
                end
            end
            ll = isempty(ll) ? l : hcat(ll, l)
        end
        states = append!(states, ll)
    end
    states
end

#add a label of 3 to avoid confussion (this state space only consists of actions)
prepare_env_states3(n) = prepare_env_states(n)

#eps greedy policy for qtable as dictionary (doesn't work this way I'm not sure why)
eps_greedy_policy(qtable, st; eps=0.1) = rand()>eps ? actions[argmax([qtable[(st,a)] for a in actions]) ] : rand(actions)

#eps greedy policy for qtable as array (more efficient and works 😄)
eps_greedy_policy2(qtable, st; eps=0.1) = rand()>eps ? actions[argmax(qtable[:,st])] : rand(actions)

#some random function that can be ignored (only works when action space is [0, pi/2])
action2angle(action) = action == 0 ? 0.0 : -pi/2

#need to use this for geting the index of an observation of the q table (can be ignored)
state_indx_f(states) = Dict(st => indx for (indx,st) in enumerate(states));

function step(state_in, ρ, σ, action, n; info = [])
    """
    Takes step in the environment
    Input:
    state_in = [±1,...,±1,0,...0, ...,0] length n
    ρ = density matrix of quantum state of dimension n x n
    σ = dm of state we want to achieve (for now dimension 2x2)
    action = some measurement basis
    n = # of qubits
    """
    state = copy(state_in)
    rw, done = 0, false
    ith_qubit = isnothing(findlast(state.!=0)) ? 1 : findlast(state.!=0) + 1
    ρ , measurement = measure_angle(ρ, action2angle(action), ith_qubit)
    state[ith_qubit] = (-1.0)^(measurement+1)
    if ith_qubit + 1 == n
        done = true
        ρ = partial_trace(ρ, 1:n-1)
        rw = reward(ρ,σ) 
    end 
    return state, ρ, rw, done, info
end


function step2(state_in, ρ, σ, action, n; info = [])
    """
    Takes step in the environment (with prepare_env_states2)
    same as step function (previous) but the state are of length 2n (measurements + actions)
    """
    state = copy(state_in)
    rw, done = 0, false
    ith_qubit = isnothing(findfirst(state.==0)) ? n : findfirst(state.==0) 
    ρ , measurement = measure_angle(ρ, action2angle(action), ith_qubit)
    state[ith_qubit] = (-1.0)^(measurement+1)
    state[ith_qubit+n] = (-1.0)^(action)
    if ith_qubit + 1 == n
        done = true
        ρ = partial_trace(ρ, 1:n-1)
        rw = reward(ρ,σ) 
        #rw = fidelity(ρ,σ)
    end 
    return state, ρ, rw, done, info
end


function step3(state_in, ρ, σ, action, n; info = [], rw_fidel=false)
    """
    Takes step in the environment (with prepare_env_states3)
    same as step function 1 but the state are actions, and we now implement byproduct op
    
    state[i] = -1 means action was pi/2
    state[i] = 1 means action was 0
    """
    state = copy(state_in)
    rw, done = 0, false
    ith_qubit = isnothing(findfirst(state.==0)) ? n : findfirst(state.==0) 
    ρ , measurement = measure_angle(ρ, action2angle(action), ith_qubit)
    state[ith_qubit] = (-1.0)^(action)
    append!(info, measurement)
    if ith_qubit + 1 == n
        done = true
        ρ = partial_trace(ρ, 1:n-1)
        #rw = reward(ρ,σ) 
#         sx = info[1] + info[3] + info[4]
#         sz = info[2] + info[3]
        sx,sz = byproduct_track(info, state)
        
        if !rw_fidel
            rw = isapprox(fidelity(apply_byproduct(ρ, sx, sz),σ), 1, rtol=0.001) ? 1 : -1 
        elseif rw_fidel
            rw = (fidelity(apply_byproduct(ρ, sx, sz),σ))
        end
    end 
    return state, ρ, rw, done, info
end


function q_learning(episodes, n, states, actions, unitary; γ=0.98, ϵ=0.1, q_table = nothing, batch=100, test_batch=0, ρ_in = nothing, σ = nothing, sarsa = false, α = 0.3, rw_fidel = false)
    """
    Implements q learning on the defined environment
    Input:
    episodes = # of episodes
    n        = # of qubits
    states   = observation space
    actions  = action space
    unitary  = the unitary gate that we want to learn
    γ        = discount factor
    eps      = for eps greedy policy
    q_table  = for using a predefined q_table
    batch    = num(ber of runs per episode (normally 1 but why not 100)
    ρ_in, σ  = for deterministic reset, if they are nothing a ψ_in will be randomly generated
    sarsa    = bool to use SARSA or Q-learning
    
    Output:
    q_table  = Q value for every (state, action) pair
    final_rewards = average reward per batch for every episode
    maxs = max reward per batch for every episode
    mins = min reward per batch for every episode
    """
    
    ep_converges = episodes
    
    #generates random state if randST is true
    randST = isnothing(ρ_in)
    if randST
        ψ = RandomUnitaryMatrix(2)*[1,0]
        σ = unitary*(ψ⊗ψ')*(unitary')    
        ρ_in =pure2density(graph_with_input(ψ, G))
    end
    ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
    
    #q_table = isnothing(q_table) ? Dict((st, act) => rand() for st in states, act in actions) : q_table
    
    #generates q table
    q_table = isnothing(q_table) ? rand(size(actions)[1], size(states)[1]) : q_table
    state_indx = state_indx_f(states)
    final_rewards = []
    maxs, mins = [],[]
    
    qbef = copy(q_table)
    
    #runs over episodes
    for eps in 1:episodes
        IJulia.clear_output(true)
        println("Episode: ", eps)
        println("Total change of Q-table: ", sum(abs.(q_table-qbef)))
        println("Q-values for state zeros(n): ", q_table[:, state_indx[zeros(n)]])
        qbef = copy(q_table)
        
        tot_rw = 0
        maxrw, minrw = -1.0, 1.0
        for j in 1:batch
            while !done
                #act = eps_greedy_policy(q_table, state)
                #println("state " ,state)
                #println(q_table[:,state_indx[state]])
                act = eps_greedy_policy2(q_table, state_indx[state]; eps=ϵ)
                state_new, ρ_new, rw, done, info = step3(state, ρ, σ, act, n, info=info, rw_fidel=rw_fidel)
                #q_table[(state, act)] = rw + γ*maximum([q_table[(state_new, a)] for a in actions])
                if !done
                    if sarsa
                        act2 = eps_greedy_policy2(q_table, state_indx[state_new], eps=ϵ)
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*q_table[act2+1, state_indx[state_new]])
                    else
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*maximum(q_table[:, state_indx[state_new]]))
                    end
                elseif done
                    q_table[act+1, state_indx[state]] = rw
                end
                state=state_new
                ρ=ρ_new
            end
            tot_rw += rw
            maxrw = maximum([maxrw, rw])
            minrw = minimum([minrw, rw])
            if randST
                ψ = RandomUnitaryMatrix(2)*[1,0]
                σ = unitary*(ψ⊗ψ')*unitary'   
                ρ_in =pure2density(graph_with_input(ψ, G))
            end
            ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
        end
        
        
        ###test qtable!
        if test_batch>0
            tot_rwt = 0 
            ψt = RandomUnitaryMatrix(2)*[1,0]
            σt = unitary*(ψt⊗ψt')*(unitary')    
            ρ_int =pure2density(graph_with_input(ψt, G))
            ρt, rwt, statet, donet, infot = copy(ρ_int), 0, zeros(n), false, []
            for j in 1:test_batch
                while !donet
                    act = eps_greedy_policy2(q_table, state_indx[statet]; eps=0)
                    statet, ρt, rwt, donet, infot = step3(statet, ρt, σt, act, n, info=infot, rw_fidel=rw_fidel)
                end
                tot_rwt += rwt
                ψt = RandomUnitaryMatrix(2)*[1,0]
                σt = unitary*(ψt⊗ψt')*unitary'   
                ρ_int =pure2density(graph_with_input(ψt, G))
                ρt, rwt, statet, donet, infot = copy(ρ_int), 0, zeros(n), false, []
            end
            
#             if isapprox(tot_rwt/test_batch,1, rtol=1e-3) 
#                 ep_converges = eps
#                 break
#             end
        end
        #saves the average over the batch size, max reward in the batch and min reward in the batch
        final_rewards =  append!(final_rewards, tot_rwt/batch)
        maxs = append!(maxs, maxrw)
        mins = append!(mins, minrw)
    end
    return q_table, (final_rewards, maxs, mins), ep_converges
end


#---------------------2D RL functions:---------
action2angle_cnot(action) = action == 0 ? 0.0 : pi/2

function this_code_is_bad(ith_qubit)
    """
    changes ith qubit to qubit in simulation for cnot
    (I will definitely write this better, really confusing with no good structure 🐛)
    """
    simq = nothing
    if ith_qubit in [1,3,5,8,10,12,14]
        simq = 1
    elseif ith_qubit in [7,2,4,6,11,13,15]
        simq = 2
    elseif ith_qubit in [9]
        simq = 3
    end
    simq    
end

function step_2d_cnot(state_in, ρ, σ, action, n; info = [], rw_fidel=false)
    """
    Takes step in the environment for cnot2d
    state[i] = -1 means action was pi/2
    state[i] = 1 means action was 0
    """
    state = copy(state_in)
    rw, done = 0, false
    ith_qubit = isnothing(findfirst(state.==0)) ? n : findfirst(state.==0) 
    
    ith_sim = this_code_is_bad(ith_qubit)
    
    n_qubs_sim = ith_qubit in [7,8,9] ? 3 : 2
    finish_cond = n_qubs_sim == 3 && ith_sim == 2 ? true : false
    
    if ith_qubit == 7
        ρ = ρ ⊗ pure2density(qubit_plus)
        ρ = swap_ij(2,3,3)*ρ*(swap_ij(2,3,3)')
        ρ = cz_after_layer_measurement(ρ,3)    
    end
    
    ρ, measurement = layer_measurement(ρ, action2angle_cnot(action), ith_sim, n_qubs_sim, finish_cond)
    
    if ith_qubit == 9
        ρ = partial_trace(ρ, [2])
    end
    
    state[ith_qubit] = (-1.0)^(action)
    append!(info, measurement)
    if ith_qubit + 2 == n
        done = true
        ## fixes the z op in output qubit 1
        ρ = arbitrary_qubit_gate(Z,1,2)*ρ*arbitrary_qubit_gate(Z,1,2)
        if !rw_fidel
            rw = isapprox(fidelity(ρ,σ), 1, rtol=0.001) ? 1 : -1 
        elseif rw_fidel
            rw = fidelity(ρ,σ)
        end
    end 
    return state, ρ, rw, done, info
end

function q_learning_cnot(episodes, n, states, actions; γ=0.98, ϵ=0.1, q_table = nothing,test_batch=0, batch=100, ρ_in = nothing, σ = nothing, sarsa = false, α = 0.3, rw_fidel = false)
    """
    Implements q learning on the defined environment
    Input:
    episodes = # of episodes
    n        = # of qubits
    states   = observation space
    actions  = action space
    γ        = discount factor
    eps      = for eps greedy policy
    q_table  = for using a predefined q_table
    batch    = num(ber of runs per episode (normally 1 but why not 100)
    ρ_in, σ  = for deterministic reset, if they are nothing a ψ_in will be randomly generated
    sarsa    = bool to use SARSA or Q-learning
    
    Output:
    q_table  = Q value for every (state, action) pair
    final_rewards = average reward per batch for every episode
    maxs = max reward per batch for every episode
    mins = min reward per batch for every episode
    """
    ep_converges = episodes
    
    #generates random state if randST is true
    randST = isnothing(ρ_in)
    if randST
        ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
        ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
        σ = pure2density(cnot_ij(1,2,2)*(ψ_in1 ⊗ ψ_in2))
        ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
    end
    ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
    
    #generates q table
    q_table = isnothing(q_table) ? rand(size(actions)[1], size(states)[1]) : q_table
    state_indx = state_indx_f(states)
    final_rewards = []
    maxs, mins = [],[]
    
    qbef = copy(q_table)
    
    #runs over episodes
    for eps in 1:episodes
        IJulia.clear_output(true)
        println("Episode: ", eps)
        println("Total change of Q-table: ", sum(abs.(q_table-qbef)))
        println("Q-values for state zeros(n): ", q_table[:, state_indx[zeros(n)]])
        qbef = copy(q_table)
        
        tot_rw = 0
        maxrw, minrw = -1.0, 1.0
        for j in 1:batch
            while !done
                act = eps_greedy_policy2(q_table, state_indx[state]; eps=ϵ)
                state_new, ρ_new, rw, done, info = step_2d_cnot(state, ρ, σ, act, n, info=info, rw_fidel=rw_fidel)
                if !done
                    if sarsa
                        act2 = eps_greedy_policy2(q_table, state_indx[state_new], eps=ϵ)
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*q_table[act2+1, state_indx[state_new]])
                    else
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*maximum(q_table[:, state_indx[state_new]]))
                    end
                elseif done
                    q_table[act+1, state_indx[state]] = rw
                end
                state=state_new
                ρ=ρ_new
            end
            tot_rw += rw
            maxrw = maximum([maxrw, rw])
            minrw = minimum([minrw, rw])
            if randST
                ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
                ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
                σ = pure2density(cnot_ij(1,2,2)*(ψ_in1 ⊗ ψ_in2))
                ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
            end
            ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
            
            
        end
        
        
        ###test qtable!
        if test_batch>0
            tot_rwt = 0 
            ψ_in1t = RandomUnitaryMatrix(2)*[0,1]
            ψ_in2t = RandomUnitaryMatrix(2)*[1,0];
            σt = pure2density(cnot_ij(1,2,2)*(ψ_in1t ⊗ ψ_in2t))
            ρ_int = pure2density(ψ_in1t ⊗ ψ_in2t);
            ρt, rwt, statet, donet, infot = copy(ρ_int), 0, zeros(n), false, []
            for j in 1:test_batch
                while !donet
                    act = eps_greedy_policy2(q_table, state_indx[statet]; eps=0)
                    statet, ρt, rwt, donet, infot = step_2d_cnot(statet, ρt, σt, act, n, info=infot, rw_fidel=rw_fidel)
                end
                tot_rwt += rwt
                ψ_in1t = RandomUnitaryMatrix(2)*[0,1]
                ψ_in2t = RandomUnitaryMatrix(2)*[1,0];
                σt = pure2density(cnot_ij(1,2,2)*(ψ_in1t ⊗ ψ_in2t))
                ρ_int = pure2density(ψ_in1t ⊗ ψ_in2t);
                ρt, rwt, statet, donet, infot = copy(ρ_int), 0, zeros(n), false, []
            end

            if isapprox(tot_rwt/test_batch,1, rtol=1e-3) 
                ep_converges = eps
                break
            end
        end

        #saves the average over the batch size, max reward in the batch and min reward in the batch
        final_rewards =  append!(final_rewards, tot_rw/batch)
        maxs = append!(maxs, maxrw)
        mins = append!(mins, minrw)
    end
        
    return q_table, (final_rewards, maxs, mins), ep_converges
end


#--------2D RL different learning curve:

function prepare_env_states_middle(n)
    """
    Prepares observation space:
    Every element is of the form
    [±1,.., ±1, 0, ...,0]
    Where the first i elements are ±1 depending on the outcome of the measurements ϕ_j j∈{1,...,i} and 0 elsewhere
    """
    outcomes = [-1,1]
    states = [zeros(n)]
    for i in 1:7
        ll = []
        for st in states[2^(i-1): end]
            cst = copy(st)
            l = []
            for o in outcomes
                new_st = copy(cst)
                new_st[i] = o
                l = isempty(l) ? [new_st] : append!(l, [new_st])
            end
            ll = isempty(ll) ? l : hcat(ll, l)
        end
        states = append!(states, ll)
    end
    states
end



function q_learning_cnot_2(episodes, n, states, actions; γ=0.98, ϵ=0.1, q_table = nothing,test_batch=0, graph_batch=10,batch=100, ρ_in = nothing, σ = nothing, sarsa = false, α = 0.3, rw_fidel = false)
    """
    Implements q learning on the defined environment
    Input:
    episodes = # of episodes
    n        = # of qubits
    states   = observation space
    actions  = action space
    γ        = discount factor
    eps      = for eps greedy policy
    q_table  = for using a predefined q_table
    batch    = num(ber of runs per episode (normally 1 but why not 100)
    ρ_in, σ  = for deterministic reset, if they are nothing a ψ_in will be randomly generated
    sarsa    = bool to use SARSA or Q-learning
    
    Output:
    q_table  = Q value for every (state, action) pair
    final_rewards = average reward per batch for every episode
    maxs = max reward per batch for every episode
    mins = min reward per batch for every episode
    """
    ep_converges = episodes
    
    #generates random state if randST is true
    randST = isnothing(ρ_in)
    if randST
        ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
        ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
        σ = pure2density(cnot_ij(1,2,2)*(ψ_in1 ⊗ ψ_in2))
        ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
    end
    ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
    
    #generates q table
    q_table = isnothing(q_table) ? rand(size(actions)[1], size(states)[1]) : q_table
    state_indx = state_indx_f(states)
    final_rewards = []
    middle_qubit_action = []
    qbef = copy(q_table)
    
    #runs over episodes
    for eps in 1:episodes
        IJulia.clear_output(true)
        println("Episode: ", eps)
        println("Total change of Q-table: ", sum(abs.(q_table-qbef)))
        println("Q-values for state zeros(n): ", q_table[:, state_indx[zeros(n)]])
        qbef = copy(q_table)
        
        tot_rw = 0
        for j in 1:batch
            while !done
                act = eps_greedy_policy2(q_table, state_indx[state]; eps=ϵ)
                state_new, ρ_new, rw, done, info = step_2d_cnot(state, ρ, σ, act, n, info=info, rw_fidel=rw_fidel)
                if !done
                    if sarsa
                        print("ERROR")
                        act2 = eps_greedy_policy2(q_table, state_indx[state_new], eps=ϵ)
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*q_table[act2+1, state_indx[state_new]])
                    else
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*maximum(q_table[:, state_indx[state_new]]))
                    end
                elseif done
                    q_table[act+1, state_indx[state]] = rw
                end
                state=state_new
                ρ=ρ_new
            end
            tot_rw += rw
            if randST
                ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
                ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
                σ = pure2density(cnot_ij(1,2,2)*(ψ_in1 ⊗ ψ_in2))
                ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
            end
            ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
            
        end
        
        
        ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
        ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
        σ = pure2density(cnot_ij(1,2,2)*(ψ_in1 ⊗ ψ_in2))
        ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
        ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
        tot_rw = 0
        k = 0
        for j in 1:graph_batch
            while !done
                if sum(state.!=0)==6
                    k+= argmax(q_table[:, state_indx[state]])-1
                end
                act = eps_greedy_policy2(q_table, state_indx[state]; eps=0)
                state, ρ, rw, done, info = step_2d_cnot(state, ρ, σ, act, n, info=info, rw_fidel=rw_fidel)
            end
            tot_rw += rw
            if randST
                ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
                ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
                σ = pure2density(cnot_ij(1,2,2)*(ψ_in1 ⊗ ψ_in2))
                ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
            end
            ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []  
        end
        
        #saves the average over the batch size, max reward in the batch and min reward in the batch
        final_rewards =  append!(final_rewards, tot_rw/graph_batch)
        
#         k = 0
#         p = 0
#         for stt in prepare_env_states_middle(15)
#             if sum(stt.!=0)==7
#                 p+=1
#                 k+= argmax(q_table[:, state_indx[stt]])-1
#             end
#         end
#         print(p)
        middle_qubit_action = append!(middle_qubit_action,k/graph_batch)
        
        ###test qtable!
        if test_batch>0
            if isapprox(tot_rw/graph_batch,1, rtol=1e-3) 
                ep_converges = eps
                break
            end
        end         
    end
        
    return q_table, final_rewards, middle_qubit_action, ep_converges
end



## Extra fuctions NOT USED-------------------------------
#operations pauli
symplectic_product( a::NamedTuple, b::NamedTuple, dim::Int) = mod(a.x'b.z - a.z'b.x, dim)
pauli_sum(a::NamedTuple, b::NamedTuple, dim::Int) = (x=mod.(a.x+b.x, dim),z=mod.(a.z+b.z, dim))



function q_learning_identity(episodes, n, states, actions; γ=0.98, ϵ=0.1, q_table = nothing,test_batch=0, graph_batch=10,batch=100, ρ_in = nothing, σ = nothing, sarsa = false, α = 0.3, rw_fidel = false)
    """
    Implements q learning on the defined environment
    Input:
    episodes = # of episodes
    n        = # of qubits
    states   = observation space
    actions  = action space
    γ        = discount factor
    eps      = for eps greedy policy
    q_table  = for using a predefined q_table
    batch    = num(ber of runs per episode (normally 1 but why not 100)
    ρ_in, σ  = for deterministic reset, if they are nothing a ψ_in will be randomly generated
    sarsa    = bool to use SARSA or Q-learning
    
    Output:
    q_table  = Q value for every (state, action) pair
    final_rewards = average reward per batch for every episode
    maxs = max reward per batch for every episode
    mins = min reward per batch for every episode
    """
    ep_converges = episodes
    
    #generates random state if randST is true
    randST = isnothing(ρ_in)
    if randST
        ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
        ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
        σ = pure2density( ψ_in1 ⊗ ψ_in2)
        ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
    end
    ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
    
    #generates q table
    q_table = isnothing(q_table) ? rand(size(actions)[1], size(states)[1]) : q_table
    state_indx = state_indx_f(states)
    final_rewards = []
    middle_qubit_action = []
    qbef = copy(q_table)
    
    #runs over episodes
    for eps in 1:episodes
        IJulia.clear_output(true)
        println("Episode: ", eps)
        println("Total change of Q-table: ", sum(abs.(q_table-qbef)))
        println("Q-values for state zeros(n): ", q_table[:, state_indx[zeros(n)]])
        qbef = copy(q_table)
        
        tot_rw = 0
        for j in 1:batch
            while !done
                act = eps_greedy_policy2(q_table, state_indx[state]; eps=ϵ)
                state_new, ρ_new, rw, done, info = step_2d_cnot(state, ρ, σ, act, n, info=info, rw_fidel=rw_fidel)
                if !done
                    if sarsa
                        print("ERROR")
                        act2 = eps_greedy_policy2(q_table, state_indx[state_new], eps=ϵ)
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*q_table[act2+1, state_indx[state_new]])
                    else
                        q_table[act+1, state_indx[state]] = (1-α)*q_table[act+1, state_indx[state]] + α*(rw + γ*maximum(q_table[:, state_indx[state_new]]))
                    end
                elseif done
                    q_table[act+1, state_indx[state]] = rw
                end
                state=state_new
                ρ=ρ_new
            end
            tot_rw += rw
            if randST
                ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
                ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
                σ = pure2density(ψ_in1 ⊗ ψ_in2);
                ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
            end
            ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
            
        end
        
        
        ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
        ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
        σ = pure2density(ψ_in1 ⊗ ψ_in2)
        ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
        ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []
        tot_rw = 0
        k = 0
        for j in 1:graph_batch
            while !done
                if sum(state.!=0)==6
                    k+= argmax(q_table[:, state_indx[state]])-1
                end
                act = eps_greedy_policy2(q_table, state_indx[state]; eps=0)
                state, ρ, rw, done, info = step_2d_cnot(state, ρ, σ, act, n, info=info, rw_fidel=rw_fidel)
            end
            tot_rw += rw
            if randST
                ψ_in1 = RandomUnitaryMatrix(2)*[0,1]
                ψ_in2 = RandomUnitaryMatrix(2)*[1,0];
                σ = pure2density(ψ_in1 ⊗ ψ_in2);
                ρ_in = pure2density(ψ_in1 ⊗ ψ_in2);
            end
            ρ, rw, state, done, info = copy(ρ_in), 0, zeros(n), false, []  
        end
        
        #saves the average over the batch size, max reward in the batch and min reward in the batch
        final_rewards =  append!(final_rewards, tot_rw/graph_batch)
        
#         k = 0
#         p = 0
#         for stt in prepare_env_states_middle(15)
#             if sum(stt.!=0)==7
#                 p+=1
#                 k+= argmax(q_table[:, state_indx[stt]])-1
#             end
#         end
#         print(p)
        middle_qubit_action = append!(middle_qubit_action,k/graph_batch)
        
        ###test qtable!
        if test_batch>0
            if isapprox(tot_rw/graph_batch,1, rtol=1e-3) 
                ep_converges = eps
                break
            end
        end         
    end
    
    return q_table, final_rewards, middle_qubit_action, ep_converges
end

u3(θ,ϕ,λ) = [cos(θ/2) -exp(im*λ)*sin(θ/2); exp(im*ϕ)*sin(θ/2) exp(im*(θ+λ))*cos(θ/2)]
rx(θ) = u3(θ,-π/2,π/2)
ry(θ) = u3(θ,0,0)

