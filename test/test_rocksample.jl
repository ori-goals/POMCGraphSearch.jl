using POMCGraphSearch
using Test
using POMDPs
using Random
using POMDPTools

# Try to load RockSample, skip tests if not available
rock_sample_available = try
    using RockSample
    true
catch e
    @warn "RockSample not available, skipping RockSample tests"
    false
end

# Set random seed for reproducibility
Random.seed!(42)

if rock_sample_available
    @testset "RockSample Basic Integration" begin
        
        # Create solver once to avoid repeated QLearning computation
        pomdp = RockSamplePOMDP(7, 8)  # Small problem for fast testing
        solver = SolverPOMCGS(pomdp;
            max_b_gap =0.3,
            max_planning_secs=60.0,  # Short time for tests
            max_search_depth=30,
            VMDP_nb_max_episode=3,
            num_sim_per_sa=10
        )
        
        @testset "Solver Creation" begin
            # Test solver creation and parameters
            @test solver isa POMDPs.Solver
            @test solver.max_b_gap == 0.3
            @test solver.max_planning_secs == 60.0
            @test solver.max_search_depth == 30
            @test solver.VMDP_nb_max_episode == 3
            @test solver.num_sim_per_sa == 10
        end
        
        @testset "Solve Function" begin
            # Test that solve runs without errors and returns policy
            policy = @test_nowarn solve(solver, pomdp)
            
            # Verify returned object is a Policy
            @test policy isa POMDPs.Policy
            @test policy isa POMCGraphSearch.FSC
            
            # Verify FSC has nodes
            @test length(policy._nodes) > 0
            @test length(policy._eta) > 0
            
            # Test action method
            action_val = @test_nowarn POMDPs.action(policy, 1)
            @test action_val isa Int
            @test action_val in actions(pomdp)
            
            # Test updater interface
            updater = @test_nowarn POMDPs.updater(policy, pomdp)
            @test updater isa POMDPs.Updater
            
            # Test belief initialization
            belief = @test_nowarn POMDPs.initialize_belief(updater, initialstate(pomdp))
            @test belief == 1  # Should start from node 1

            # Test simulation interface
            updater_instance = POMDPs.updater(policy, pomdp)
            
            # Test basic simulation
            history = @test_nowarn simulate(
                HistoryRecorder(max_steps=10),
                pomdp,
                policy,
                updater_instance,
                1  # Start from node 1
            )
            
            # Verify simulation results
            @test length(history) > 0
            @test discounted_reward(history) isa Float64
            @test isfinite(discounted_reward(history))

            # Check FSC structure
            @test length(policy._nodes_VQMDP_labels) == length(policy._nodes)
        end
        
    end
    
    println("✓ Basic RockSample integration tests passed!")
    
else
    @testset "RockSample Tests" begin
        @test_skip "RockSample package not available"
    end
    println("⚠ RockSample tests skipped - package not available")
end