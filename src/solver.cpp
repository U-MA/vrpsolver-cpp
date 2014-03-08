#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "mct_node.h"
#include "mct_selector.h"
#include "simulator.h"
#include "node.h"
#include "solver.h"
#include "vehicle_manager.h"

void Solver::setProblem(const char *filename)
{
    vrp_.Create(filename);
    printf("file name       : %s\n", filename);
}

void Solver::setSeed(long seed)
{
    this->seed_ = seed;
}

void Solver::setMctsIterationCount(int count)
{
    this->count_ = count;
}

void Solver::setSimulationCount(int count)
{
    this->simulation_count_ = count;
}

void Solver::printRunParameter(void)
{
    printf("seed            : %ld\n"  , seed_);
    printf("search count    : %d\n"   , count_);
    printf("simulation count: %d\n\n" , simulation_count_);
}


/* モンテカルロ木探索の実行 */
void Solver::run(void)
{
    printRunParameter();
    unsigned int threshold = 4;

    Solution solution(vrp_);
    while (!solution.IsFinish())
    {
        MctNode root(0);

        /*
        for (int i=0; i < count_; i++)
            mct.build(vrp_, vm, simulation_count_);
            */
        for (int i=0; i < count_; i++)
        {
            std::vector<MctNode*> visited;
            // Selection
            MctNode *node = Selector::Ucb(root, visited);

            Solution solution_copy;
            solution.Copy(solution_copy);

            for (unsigned int i=1; i < visited.size(); i++)
            {
                int move = visited[i]->CustomerId();
                if (move != 0)
                    solution_copy.CurrentVehicle()->Visit(vrp_, move);
                else
                    solution_copy.ChangeVehicle();
            }

            // Expansion
            if (!solution_copy.IsFinish() && (node->Count() >= threshold))
            {
                for (unsigned int j=0; j <= vrp_.CustomerSize(); j++)
                {
                    if (!solution_copy.IsVisit(j) &&
                        (solution_copy.CurrentVehicle()->Capacity() + vrp_.Demand(j) <= vrp_.Capacity()))
                    {
                        node->CreateChild(j);
                    }
                }
                if (solution_copy.CurrentVehicleId()+1 < vrp_.VehicleSize())
                {
                    node->CreateChild(0);
                }
                visited.pop_back();
                node = Selector::Ucb(*node, visited);

                int move = (*visited.rbegin())->CustomerId();
                if (move != 0)
                    solution_copy.CurrentVehicle()->Visit(vrp_, move);
                else
                    solution_copy.ChangeVehicle();

            }

            // Simulation
            Simulator simulator;
            unsigned int cost = simulator.sequentialRandomSimulation(vrp_, solution_copy);
            solution_copy.Print();
            int tmp_cost = (int)cost;

            // Backpropagation
            for (unsigned int i=0; i < visited.size(); i++)
            {
                visited[i]->Update(-tmp_cost);
            }

        }

        long max_value = -10000000;
        MctNode *next = NULL;
        for (unsigned int i=0; i < root.ChildSize(); i++)
        {
            if (root.Child(i)->Value() > max_value)
            {
                max_value = root.Child(i)->Value();
                next = root.Child(i);
            }
        }
        if (next->CustomerId() != 0)
            solution.CurrentVehicle()->Visit(vrp_, next->CustomerId());
        else
            solution.ChangeVehicle();
    }

    solution.Print();
    printf("[COST] %6d\n", solution.ComputeTotalCost(vrp_));
}
