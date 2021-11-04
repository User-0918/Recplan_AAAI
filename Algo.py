from Problem import Problem, Node


CUTOFF = 0
FAILURE = -1

        
def Iterative_Deepening_Search(problem):
    # Call DLS using different depth limits

    def Depth_Limited_Search(problem, limit):
        # Initialize and call Recursive_DLS 
        
        def Recursive_DLS(node, problem, limit):
            # Implement Depth Limited Search in a recusive way
            # CUTOFF: No solution (within the given limit)
            # FAILURE: No solution (within the whole tree)
            if problem.GoalTest(node.state):
                # print('success')
                return node.solution()
            elif limit == 0:
                return CUTOFF
            else:
                cutoff_occurred = False
                for action in problem.Actions(node.state):
                    child = node.child_node(problem, action)
                    result = Recursive_DLS(child, problem, limit-1)
                    if result == CUTOFF:
                        cutoff_occurred = True
                    elif result != FAILURE:
                        return result
                if cutoff_occurred:
                    return CUTOFF
                else:
                    return FAILURE

        node = Node(problem.initial_state)
        return Recursive_DLS(node, problem, limit)

    depth = 0
    while True:
        result = Depth_Limited_Search(problem, depth)
        if result != CUTOFF and result != FAILURE:
            return result
        depth += 1


def RE_Greedy_Search(problem):
    # Initialize and call Recursive_GS

    def Recursive_GS(node, problem):
        # Implement Greedy Search in a recursive way
        if problem.GoalTest(node.state):
            return node.solution()
        successors = []
        for action in problem.Actions(node.state):
            child = node.child_node(problem, action)
            successors.append(child)
        if len(successors) == 0:
            return FAILURE
        for s in successors:
            s.f = problem.h1(s.state)           # evaluation value = the number of misplaced tiles
        while True:
            successors.sort(key=lambda x: x.f)  # sort the all the nodes by evaluation values
            best = successors[0]                # choose the node with lowest evaluation value
            result = Recursive_GS(best, problem)
            if result != FAILURE:
                return result

    node = Node(problem.initial_state)
    result = Recursive_GS(node, problem)
    return result


def Greedy_Search(problem):
    # Implement Greedy Search in a non-recursive way
    node = Node(problem.initial_state)
    successors = []         # list containing all possible "next states"
    successors.append(node)
    explored = set()        # to record situations that have already appeared
    while len(successors) > 0:
        node = successors.pop(0)    # choose the node with lowest evaluation value
        if problem.GoalTest(node.state):
            return node.solution()
        explored.add(node.state)
        for action in problem.Actions(node.state):
            child = node.child_node(problem, action)
            if child not in successors and child.state not in explored:
                successors.append(child)
                child.f = problem.h1(child.state)   # evaluation value = the number of misplaced tiles
            successors.sort(key=lambda x: x.f)

    return FAILURE


def RE_Astar_Search(problem):
    # Initialize and call Recusive_AS

    def Recursive_AS(node, problem):
        # Implement A* Search in a recursive way
        if problem.GoalTest(node.state):
            return node.solution()
        successors = []
        for action in problem.Actions(node.state):
            child = node.child_node(problem, action)
            successors.append(child)
        if len(successors) == 0:
            return FAILURE
        for s in successors:
            s.f = s.depth + problem.h(s.state)  # f(n) = g(n) + h(n), where h(n) = max(h1(n), h2(n))
        while True:
            successors.sort(key=lambda x: x.f)  # sort the all the nodes by evaluation values
            best = successors[0]                # choose the node with lowest evaluation value
            # print(best.depth)
            result = Recursive_AS(best, problem)
            if result != FAILURE:
                return result

    node = Node(problem.initial_state)
    result = Recursive_AS(node, problem)
    return result


def Astar_Search(problem):
    # Implement A* Search in a non-recursive way
    node = Node(problem.initial_state)
    successors = []         # list containing all possible "next states"
    successors.append(node)
    explored = set()        # to record situations that have already appeared
    while len(successors) > 0:
        node = successors.pop(0)
        if problem.GoalTest(node.state):
            return node.solution()
        explored.add(node.state)
        for action in problem.Actions(node.state):
            child = node.child_node(problem, action)
            if child not in successors and child.state not in explored:
                successors.append(child)
                child.f = child.depth + problem.h(child.state)  # f(n) = g(n) + h(n), where h(n) = max(h1(n), h2(n))
            successors.sort(key=lambda x: x.f)

    return FAILURE
