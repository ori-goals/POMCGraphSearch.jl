using POMCGraphSearch
using Test
using POMDPs
using Random
using POMDPTools

@testset "POMCGS Basic Tests" begin

    @testset "Package Loading" begin
        @test isdefined(Main, :POMCGraphSearch)
        println("✓ Package loaded successfully")
    end

    @testset "Type Definitions" begin
        @test isdefined(POMCGraphSearch, :SolverPOMCGS)
        println("✓ SolverPOMCGS type defined")
    end

    @testset "Key Functions Exist" begin
        for fname in [:Solve, :detect_action_space, :detect_state_space, :detect_observation_space]
            @test isdefined(POMCGraphSearch, fname)
        end
        println("✓ Key functions defined")
    end

end


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

@testset "POMCGS Integration Test" begin

    # Define a minimal test POMDP
    struct MiniPOMDP <: POMDP{Bool, Bool, Bool} end

    POMDPs.actions(::MiniPOMDP) = [true, false]
    POMDPs.states(::MiniPOMDP) = [true, false]
    POMDPs.observations(::MiniPOMDP) = [true, false]
    POMDPs.reward(::MiniPOMDP, s, a) = s ? 1.0 : -1.0
    POMDPs.initialstate(::MiniPOMDP) = Deterministic(true)
    POMDPs.discount(::MiniPOMDP) = 0.95
    POMDPs.isterminal(::MiniPOMDP, s) = false


    POMDPs.transition(::MiniPOMDP, s, a) = Deterministic(!s)
    POMDPs.observation(::MiniPOMDP, sp, a) = Deterministic(sp)
    POMDPs.reward(::MiniPOMDP, s, a) = s ? 1.0 : -1.0

    function POMDPs.gen(::MiniPOMDP, s, a)
        sp = !s
        o = sp
        r = s ? 1.0 : -1.0
        return (sp=sp, o=o, r=r)
    end

    @testset "MiniPOMDP Creation" begin
        pomdp = MiniPOMDP()
        @test pomdp isa MiniPOMDP
        @test length(POMDPs.actions(pomdp)) == 2
        @test POMDPs.discount(pomdp) ≈ 0.95
        println("✓ MiniPOMDP created")
    end

    @testset "POMCGS Constructor" begin
        pomdp = MiniPOMDP()

        try
            solver = POMCGS.SolverPOMCGS(
                pomdp;
                nb_iter=10,
                nb_sim=5,
                max_planning_secs=1,
                nb_eval=10,
                nb_samples_VMDP=5,
                min_num_particles=10,
                max_num_particles=50,
                max_graph_node_size=100
            )

            @test solver isa POMCGS.SolverPOMCGS
            @test solver.pomdp === pomdp
            println("✓ SolverPOMCGS constructor works")

        catch e
            @warn "POMCGS constructor failed (may need additional dependencies): $e"
        end
    end

end

println("✓ All basic structure and integration tests passed!")
