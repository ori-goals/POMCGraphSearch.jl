using POMCGraphSearch
using Test
using POMDPs
using Random
using POMDPTools

# Try to load POMDPModels, skip tests if not available
lightdark_available = try
    using POMDPModels
    true
catch e
    @warn "POMDPModels not available, skipping LightDark tests"
    false
end

# Set random seed for reproducibility
Random.seed!(42)

if lightdark_available
    @testset "LightDark1D Basic Integration" begin
        
        # Create solver once to avoid repeated computation
        pomdp = LightDark1D()
        solver = SolverPOMCGS(pomdp;
            max_b_gap=0.2,
            max_planning_secs=60.0, 
            max_search_depth=30,
            num_fixed_observations=20,
            state_grid = [1.0, 1.0],
            num_sim_per_sa=500
        )
        
        @testset "Solver Creation" begin
            # Test solver creation and parameters
            @test solver isa POMDPs.Solver
            @test solver.max_b_gap == 0.2
            @test solver.max_planning_secs == 60.0
            @test solver.max_search_depth == 30
            @test solver.num_fixed_observations == 20
            @test solver.state_grid == [1.0, 1.0]
            @test solver.num_sim_per_sa == 500
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
            updater = @test_nowarn POMDPs.updater(policy)
            @test updater isa POMDPs.Updater
            
            # Test belief initialization
            belief = @test_nowarn POMDPs.initialize_belief(updater, initialstate(pomdp))
            @test belief == 1  # Should start from node 1

            # Test simulation interface
            updater_instance = POMDPs.updater(policy)
            
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
            
            # Test multiple nodes have actions
            for node_id in 1:min(3, length(policy._nodes))
                action_val = POMDPs.action(policy, node_id)
                @test action_val in [-1, 0, 1]  # LightDark actions: left, stay, right
            end
            
            # Test belief update with mock observation
            updater = POMDPs.updater(policy)
            current_node = 1
            action = POMDPs.action(policy, current_node)
            test_observation = 0.5  # Example observation for LightDark
            
            next_node = @test_nowarn POMDPs.update(updater, current_node, action, test_observation)
            @test next_node isa Int
            @test next_node == -1 || (1 <= next_node <= length(policy._nodes))
        end
        
    end
    
    println("✓ Basic LightDark1D integration tests passed!")
    
else
    @testset "LightDark Tests" begin
        @test_skip "POMDPModels not available"
    end
    println("⚠ LightDark tests skipped - POMDPModels not available")
end