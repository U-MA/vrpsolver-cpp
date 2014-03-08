#include "CppUTest/TestHarness.h"

#include <iostream>
#include <vector>

#include "host_vrp.h"
#include "mct_node.h"
#include "mct_selector.h"
#include "simulator.h"

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
            cout << "SELECTION@" << &root << endl;
            MctNode *node = Selector::Ucb(root, visited);

            Solution solution_copy;
            solution.Copy(solution_copy);

            for (unsigned int i=1; i < visited.size(); i++)
            {
                int move = visited[i]->CustomerId();
                cout << "MOVE: " << move << endl;
                if (move != 0)
                    solution_copy.CurrentVehicle()->Visit(host_vrp, move);
                else
                    solution_copy.ChangeVehicle();
            }

            // Expansion
            if (!solution_copy.IsFinish() && (node->Count() >= threshold))
            {
                cout << "EXPANSION@" << node << "(MOVE " << node->CustomerId() << ")" << endl;
                for (unsigned int j=0; j <= host_vrp.CustomerSize(); j++)
                {
                    if (!solution_copy.IsVisit(j) &&
                        (solution_copy.CurrentVehicle()->Capacity() + host_vrp.Demand(j) <= host_vrp.Capacity()))
                    {
                        cout << "CUSTOMER " << j << " EXPAND" << endl;
                        node->CreateChild(j);
                    }
                }
                if (solution_copy.CurrentVehicleId()+1 < host_vrp.VehicleSize())
                {
                    cout << "NEXT VEHICLE EXPAND" << endl;
                    node->CreateChild(0);
                }
                visited.pop_back();
                node = Selector::Ucb(*node, visited);

                int move = (*visited.rbegin())->CustomerId();
                cout << "MOVE: " << move << endl;
                if (move != 0)
                    solution_copy.CurrentVehicle()->Visit(host_vrp, move);
                else
                    solution_copy.ChangeVehicle();

            }

            // Simulation
            Simulator simulator;
            cout << "SIMULATION" << endl;
            unsigned int cost = simulator.sequentialRandomSimulation(host_vrp, solution_copy);
            solution_copy.Print();
            cout << "COST " << cost << endl;
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
        cout << "NEXT MOVE IS " << next->CustomerId() << endl;
        if (next->CustomerId() != 0)
            solution.CurrentVehicle()->Visit(host_vrp, next->CustomerId());
        else
            solution.ChangeVehicle();
    }

    int cost;
    if (solution.IsFeasible())
        cost = solution.ComputeTotalCost(host_vrp);
    else
        cost = 10000;

    std::cout << "cost:" << cost << std::endl;
    CHECK(cost < 10000);
}
