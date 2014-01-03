#include "CppUTest/TestHarness.h"

#include "Node.h"


TEST_GROUP(Node)
{
};

IGNORE_TEST(Node, start)
{
    FAIL("fail");
}
