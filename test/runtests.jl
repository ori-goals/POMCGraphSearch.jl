# test/runtests.jl

using POMCGraphSearch
using Test

using POMCGraphSearch
using Test

# Detect environment
is_ci = get(ENV, "CI", "false") == "true"
run_full_tests = !is_ci && get(ENV, "POMCGS_FULL_TEST", "false") == "true"

println("POMCGraphSearch Test Suite")
println("===========================")
println("CI Environment: $is_ci")
println("Full Tests: $run_full_tests")

# Essential tests - always run
println("\n1. Running essential tests...")
include("basic_structure_tests.jl")

# Heavy integration tests - conditional
if run_full_tests
    println("\n2. Running RockSample integration...")
    include("test_rocksample.jl")
    
    println("\n3. Running LightDark integration...")
    include("test_lightdark.jl")
else
    println("\n2. Skipping heavy integration tests")
    println("   Set POMCGS_FULL_TEST=true to enable full test suite, e.g., in Julia REPL run: ENV[\"POMCGS_FULL_TEST\"] = \"true\"")
    @test_skip "Heavy tests disabled"
end

println("\n✓ Test suite completed!")