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
        for fname in [:Solve, :solve, :action, :update]
            @test isdefined(POMCGraphSearch, fname)
        end
        println("✓ Key functions defined")
    end

end


println("✓ All basic structure tests passed!")
