#include "CppUTest/TestHarness.h"

#include <iostream>
#include <vector>

#include "host_vrp.h"
#include "mct_node.h"
#include "mct_selector.h"
#include "simulator.h"

TEST_GROUP(Mcts)
{
};

TEST(Mcts, E_n13_k4)
{
    HostVrp host_vrp("Vrp-All/E/E-n13-k4.vrp");
    Solution solution;
    const unsigned int threshold = 1;

    while (!solution.IsFinish())
    {
        MctNode root(0);
        std::vector<MctNode*> visited;
        for (int i=0; i < 1000; i++)
        {
            // Selection
            MctNode *node = Selector::Ucb(root, visited);

            // Expansion
            if (node->Count() >= threshold)
            {
                for (int j=0; j < host_vrp.customer_size(); j++)
                {
                    if (solution.IsVisit(j))
                        node->CreateChild(j);
                }
                // 次の車体があれば子に追加
                node = Selector::Ucb(*node, visited);
            }

            // solutionの更新
            for (std::vector<MctNode*>::iterator iter = visited.begin();
                 iter != visited.end(); iter++)
            {
                int move = (*iter)->CustomerId();
                if (move != 0)
                    solution.CurrentVehicle()->visit(host_vrp, move);
                else
                    solution.ChangeVehicle();
            }

            // Simulation
            Simulator simulator;
            unsigned int cost = simulator.random(host_vrp, solution);

            // Backpropagation
            for (std::vector<MctNode*>::iterator iter = visited.begin();
                 iter != visited.end(); iter++)
            {
                (*iter)->Update(cost);
            }
        }
    }

    int cost;
    if (solution.IsFeasible())
        cost = solution.ComputeTotalCost(host_vrp);
    else
        cost = 10000;

    std::cout << "cost:" << cost << std::endl;
    CHECK(cost < 10000);
}
