#include "CppUTest/TestHarness.h"

#include <iostream>
#include <vector>
#include <limits.h>

#include "host_vrp.h"
#include "mct_node.h"
#include "mct_selector.h"
#include "simulator.h"
#include "solution.h"
#include "solution_helper.h"

using namespace std;

TEST_GROUP(Mcts)
{
    void setup()
    {
        srand(2013);
    }
};

TEST(Mcts, E_n13_k4)
{
    HostVrp host_vrp("Vrp-All/E/E-n13-k4.vrp");
    Solution solution(host_vrp);
    const unsigned int threshold = 4;

    while (!solution.IsFinish())
    {
        MctNode root(0);
        for (int i=0; i < 1000; i++)
        {
            std::vector<MctNode*> visited;

            // Selection
            MctNode *node = Selector::Ucb(root, visited);

            Solution solution_copy;
            solution.Copy(solution_copy);

            for (unsigned int i=1; i < visited.size(); i++)
                SolutionHelper::Transition(solution_copy, host_vrp, visited[i]->CustomerId());

            // Expansion
            if (!solution_copy.IsFinish() && (node->Count() >= threshold))
            {
                for (unsigned int j=0; j <= host_vrp.CustomerSize(); j++)
                {
                    if (!solution_copy.IsVisit(j) &&
                        (solution_copy.CurrentVehicle()->Capacity() + host_vrp.Demand(j) <= host_vrp.Capacity()))
                        node->CreateChild(j);
                }
                if (solution_copy.CurrentVehicleId()+1 < host_vrp.VehicleSize())
                    node->CreateChild(0);

                visited.pop_back();
                node = Selector::Ucb(*node, visited);

                int move = (*visited.rbegin())->CustomerId();
                SolutionHelper::Transition(solution_copy, host_vrp, move);
            }

            // Simulation
            Simulator simulator;
            unsigned int cost = simulator.sequentialRandomSimulation(host_vrp, solution_copy);
            int tmp_cost = (int)cost;

            // Backpropagation
            for (unsigned int i=0; i < visited.size(); i++)
                visited[i]->Update(-tmp_cost);
        }

        double max_ave_value = -1000000;
        MctNode *next = NULL;
        for (unsigned int i=0; i < root.ChildSize(); i++)
        {
            double ave_value = (double)root.Child(i)->Value() / root.Child(i)->Count();
            if (ave_value > max_ave_value)
            {
                max_ave_value = ave_value;
                next = root.Child(i);
            }
        }
        SolutionHelper::Transition(solution, host_vrp, next->CustomerId());
    }

    int cost;
    if (solution.IsFeasible())
        cost = solution.ComputeTotalCost(host_vrp);
    else
        cost = 10000;

    std::cout << "COST:" << cost << std::endl;
    CHECK(cost < 10000);
}

TEST(Mcts, E_n51_k5)
{
    HostVrp host_vrp("Vrp-All/E/E-n51-k5.vrp");
    Solution solution(host_vrp);
    const unsigned int threshold = 4;

    while (!solution.IsFinish())
    {
        MctNode root(0);
        for (int i=0; i < 1000; i++)
        {
            std::vector<MctNode*> visited;

            // Selection
            MctNode *node = Selector::Ucb(root, visited);

            Solution solution_copy;
            solution.Copy(solution_copy);

            for (unsigned int j=1; j < visited.size(); j++)
                SolutionHelper::Transition(solution_copy, host_vrp, visited[j]->CustomerId());

            // Expansion
            if (!solution_copy.IsFinish() && (node->Count() >= threshold))
            {
                for (unsigned int j=0; j <= host_vrp.CustomerSize(); j++)
                {
                    if (!solution_copy.IsVisit(j) &&
                        (solution_copy.CurrentVehicle()->Capacity() + host_vrp.Demand(j) <= host_vrp.Capacity()))
                        node->CreateChild(j);
                }
                if (solution_copy.CurrentVehicleId()+1 < host_vrp.VehicleSize())
                    node->CreateChild(0);

                visited.pop_back();
                node = Selector::Ucb(*node, visited);

                int move = (*visited.rbegin())->CustomerId();
                SolutionHelper::Transition(solution_copy, host_vrp, move);
            }

            // Simulation
            Simulator simulator;
            unsigned int cost = simulator.sequentialRandomSimulation(host_vrp, solution_copy);
            int tmp_cost = (int)cost;

            // Backpropagation
            for (unsigned int j=0; j < visited.size(); j++)
                visited[j]->Update(-tmp_cost);
        }

        double max_ave_value = -1000000;
        MctNode *next = NULL;
        for (unsigned int i=0; i < root.ChildSize(); i++)
        {
            double ave_value = (double)root.Child(i)->Value() / root.Child(i)->Count();
            if (ave_value > max_ave_value)
            {
                max_ave_value = ave_value;
                next = root.Child(i);
            }
        }
        SolutionHelper::Transition(solution, host_vrp, next->CustomerId());
    }

    int cost;
    if (solution.IsFeasible())
        cost = solution.ComputeTotalCost(host_vrp);
    else
        cost = 10000;

    std::cout << "FINAL COST:" << cost << std::endl;
    CHECK(cost < 10000);
}

TEST(Mcts, E_n101_k14)
{
    HostVrp host_vrp("Vrp-All/E/E-n101-k14.vrp");
    Solution solution(host_vrp);
    const unsigned int threshold = 4;

    while (!solution.IsFinish())
    {
        MctNode root(0);
        for (int i=0; i < 1000; i++)
        {
            std::vector<MctNode*> visited;

            // Selection
            MctNode *node = Selector::Ucb(root, visited);

            Solution solution_copy;
            solution.Copy(solution_copy);

            for (unsigned int j=1; j < visited.size(); j++)
                SolutionHelper::Transition(solution_copy, host_vrp, visited[j]->CustomerId());

            // Expansion
            if (!solution_copy.IsFinish() && (node->Count() >= threshold))
            {
                for (unsigned int j=0; j <= host_vrp.CustomerSize(); j++)
                {
                    if (!solution_copy.IsVisit(j) &&
                        (solution_copy.CurrentVehicle()->Capacity() + host_vrp.Demand(j) <= host_vrp.Capacity()))
                        node->CreateChild(j);
                }
                if (solution_copy.CurrentVehicleId()+1 < host_vrp.VehicleSize())
                    node->CreateChild(0);

                visited.pop_back();
                node = Selector::Ucb(*node, visited);

                int move = (*visited.rbegin())->CustomerId();
                SolutionHelper::Transition(solution_copy, host_vrp, move);
            }

            // Simulation
            Simulator simulator;
            unsigned int cost = simulator.sequentialRandomSimulation(host_vrp, solution_copy);
            int tmp_cost = (int)cost;

            // Backpropagation
            for (unsigned int j=0; j < visited.size(); j++)
                visited[j]->Update(-tmp_cost);
        }

        double max_ave_value = -1000000;
        MctNode *next = NULL;
        for (unsigned int i=0; i < root.ChildSize(); i++)
        {
            double ave_value = (double)root.Child(i)->Value() / root.Child(i)->Count();
            if (ave_value > max_ave_value)
            {
                max_ave_value = ave_value;
                next = root.Child(i);
            }
        }
        SolutionHelper::Transition(solution, host_vrp, next->CustomerId());
    }

    int cost;
    if (solution.IsFeasible())
        cost = solution.ComputeTotalCost(host_vrp);
    else
        cost = 10000;

    std::cout << "FINAL COST:" << cost << std::endl;
    CHECK(cost < 10000);
}
