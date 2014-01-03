#ifndef VRPSOLVER_CPP_NODE_H
#define VRPSOLVER_CPP_NODE_H

class Node
{
private:
    int count_;
    int childSize_;

public:
    Node(void);
    ~Node(void);

    int count(void) const;
    int childSize(void) const;
    void expand(int childSize);
    Node *select(void);
};

#endif /* VRPSOLVER_CPP_NODE_H */
