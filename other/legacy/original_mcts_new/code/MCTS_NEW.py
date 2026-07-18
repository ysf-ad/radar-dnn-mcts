import numpy as np
import copy
import gc

# import pickle
# import os
# import math
# import random
# import hashlib


class Node:

    # initialize node class
    def __init__(self, args, problem_input, parent=None):
        self.args = args
        self.problem_input = problem_input

        self.visits = 0
        self.children = []
        self.parent = parent
        self.terminal = False
        self.found_terminal = False
        self.complete = False  # If all nodes below have been visited
        self.expanded = False
        self.time = 0
        self.best_cost = self.args["max_cost"]  # best cost found by this node
        self.child_id = []  # the child number w.r.t parent
        self.task_id = []
        self.best_child = []  # the child_id that has found the best cost
        self.task_id_order_so_far = []  # order of id's scheduled so far
        self.task_id_order_best = []  # best complete order of scheduled id's
        self.dropped_id_order_best = []  # best complete order of dropped id's

    def add_child(self, child_problem):
        child = Node(self.args, child_problem, self)
        self.children.append(child)

    def is_terminal(self):
        if np.sum(self.problem_input[0, 6, :, 0]) + np.sum(self.problem_input[0, 7, :, 0])\
                == self.args["NumTasks"]:
            return True
        else:
            return False

    def calc_cost(self):
        start_time = self.problem_input[0, 0, :, 0]  # * self.args["Total_time"]
        tardiness_cost = self.problem_input[0, 2, :, 0]  # (self.problem_input[0, 2, :, 0] * 4) + 1
        drop_cost = self.problem_input[0, 4, :, 0]  # (self.problem_input[0, 4, :, 0] * 400) + 100
        exec_time = self.problem_input[0, 5, :, 0]  # * self.args["Total_time"]
        scheduled = self.problem_input[0, 6, :, 0]
        dropped = self.problem_input[0, 7, :, 0]

        sched_active = self.problem_input[0, 6, :, 0] + self.problem_input[0, 7, :, 0]
        action_vector = np.asarray(-sched_active + 1)
        unscheduled_cost = np.sum(np.multiply(action_vector, drop_cost))

        delays = exec_time - start_time  # calculate all delays
        delays_masked = np.multiply(scheduled, delays)  # mask delays
        delays_cost = np.sum(np.multiply(tardiness_cost, delays_masked))  # mask delays

        drops_cost = np.sum(np.multiply(dropped, drop_cost))  # drop cost
        total_cost = delays_cost + drops_cost + unscheduled_cost

        return total_cost

    def check_dropped(self):
        # flip zeros and ones since zero means valid action
        sched_active = self.problem_input[0, 6, :, 0] + self.problem_input[0, 7, :, 0]
        action_vector = np.asarray(-sched_active + 1)
        action_index = np.asarray([i for i in range(action_vector.size) if action_vector[i] == 1])
        node_time = self.time # * self.args["Total_time"]
        while np.sum(action_vector) != 0:
            start_time = self.problem_input[0, 0, action_index[0], 0] # * self.args["Total_time"]
            drop_time = self.problem_input[0, 3, action_index[0], 0] # 2 + (110 * self.problem_input[0, 3, action_index[0], 0])
            exec_time = max(start_time, node_time)
            if exec_time > self.args["Total_time"] or exec_time > drop_time:
                self.problem_input[0, 7, action_index[0], 0] = 1  # i'th task dropped and change node.problem_input
            action_vector[action_index[0]] = 0
            action_index = action_index[1:]
        return

    def check_complete(self):
        num_children = len(self.children)
        if self.terminal:
            return True
        # elif num_children == 0:
        #     return False
        else:
            children_complete = np.zeros(num_children)
            for i in range(num_children):
                children_complete[i] = self.children[i].complete
            if np.sum(children_complete) == num_children:
                return True
            else:
                return False

    def get_valid_actions(self):
        # flip zeros and ones since zero means valid action
        sched_active = self.problem_input[0, 6, :, 0] + self.problem_input[0, 7, :, 0]
        action_vector = np.asarray(-sched_active + 1)
        return action_vector


class MCTS:

    def __init__(self, args):
        self.args = args
        self.min_cost = self.args["max_cost"]
        self.node_instances = []


    # This is the main loop of the MCTS, it is to return the best node after NumSim MCTS simulations
    # It goes forward until a terminal node in a single simulation and then back propagates all
    # important information until it goes back to the root node and then after NumSim MCTS simulations
    # it selects the first task of the task order that produced the minimum cost
    def search(self, node_start, FindOptimal):
        i = 0
        while i < self.args["NumMCTSRollouts"]:
            leaf_node = self.tree_policy(node_start)
            self.back_up(leaf_node)
            if FindOptimal:
                i -= 1
            if node_start.complete is True or node_start.best_cost == 0:
                break
            i += 1
        self.node_instances.append(node_start)
        return

    # This is the method that iteratively selects children nodes until it finds a leaf node

    # def tree_policy(self, node):
    #     while node.expanded and not node.terminal:
    #         node = self.policy_select(node)
    #     if not node.terminal:
    #         self.expand(node)
    #         # node = self.policy_select(node)
    #     return node

    def tree_policy(self, node):
        # while node.expanded and not node.terminal:
        while not node.terminal:
            if not node.expanded:
                self.expand(node)
            node = self.policy_select(node)

        return node

    def expand(self, node):
        action_vector = node.get_valid_actions()
        action_index = np.asarray([i for i in range(action_vector.size) if action_vector[i] == 1])
        node.expanded = True
        child_counter = 0
        while np.sum(action_vector) != 0:
            # create child problem
            child_problem_copy = copy.deepcopy(node.problem_input)  # copy of child problem
            node_time = node.time # * self.args["Total_time"]
            start_time = node.problem_input[0, 0, action_index[0], 0] # * self.args["Total_time"]
            task_interval = node.problem_input[0, 1, action_index[0], 0] # 2 + (10 * node.problem_input[0, 1, action_index[0], 0])
            drop_time = node.problem_input[0, 3, action_index[0], 0] # 2 + (110 * node.problem_input[0, 3, action_index[0], 0])
            exec_time = max(start_time, node_time)
            # if exec_time < self.args["Total_time"] and exec_time < drop_time:
                # node.problem_input[0, 7, action_index[0], 0] = 1
                # child_counter -= 1
            # else:
            child_problem_copy[0, 5, action_index[0], 0] = exec_time # / self.args["Total_time"]  # execution time
            child_problem_copy[0, 6, action_index[0], 0] = 1  # i'th task scheduled
            # add and update child node
            node.add_child(child_problem_copy)  # add new child node for action
            node.children[-1].time = (exec_time + task_interval) # / self.args["Total_time"]  # node time after
            node.children[-1].task_id = action_index[0]
            node.children[-1].check_dropped()  # check for any dropped tasks
            node.children[-1].terminal = node.children[-1].is_terminal()
            # if node.children[-1].is_terminal():
            #     node.children[-1].terminal = True
            #     node.children[-1].expanded = True  # check if terminal node
            #     node.children[-1].found_terminal = True
            # node.children[-1].complete = node.children[-1].check_complete()
            node.children[-1].best_cost = node.children[-1].calc_cost()  # calculate node cost
            node.children[-1].task_id_order_so_far = \
                np.append(node.task_id_order_so_far, action_index[0])  # task order id's scheduled so far
            node.children[-1].child_id = child_counter

            # update action vector and index
            action_vector[action_index[0]] = 0
            action_index = action_index[1:]
            child_counter += 1
        return

    def policy_select(self, node):
        valid_actions = node.get_valid_actions()
        action_index = np.asarray([i for i in range(valid_actions.size) if valid_actions[i] == 1])

        complete_nodes = np.ones(action_index.shape)
        num_visit = np.zeros(action_index.shape)
        cost_node = np.zeros(action_index.shape)
        for i in range(len(node.children)):
            num_visit[i] = node.children[i].visits
            cost_node[i] = node.children[i].best_cost
            if node.children[i].complete is True:
                complete_nodes[i] = 0

        # uniform distribution
        prob_masked = np.ones(action_index.size) / action_index.size
        util = prob_masked
        util = np.multiply(util, complete_nodes)
        if np.sum(util) == 0:
            util = np.multiply(np.ones(util.size), complete_nodes) / np.sum(complete_nodes)
        else:
            util = util / np.sum(util)

        a = int(np.random.choice(action_index.size, 1, p=np.ndarray.flatten(util)))

        node = node.children[a]
        return node

    def back_up(self, node):
        was_terminal = False
        leaf_cost = node.best_cost
        leaf_order = node.task_id_order_so_far
        if node.terminal is True:
            dropped = node.problem_input[0, 7, :, 0]
            terminal_dropped_index = np.asarray([i for i in range(dropped.size) if dropped[i] == 1])
            was_terminal = True
        level_count = 0  # level above leaf node
        while node is not None:
            node.visits += 1
            node.complete = node.check_complete()
            if node.complete:
                node.children = []
            if leaf_cost <= node.best_cost:
                node.best_cost = leaf_cost
                if was_terminal is True:
                    node.dropped_id_order_best = terminal_dropped_index
                    node.task_id_order_best = leaf_order
                    node.found_terminal = True
                    # Only store training data of nodes along paths with terminal node
                    if level_count == 0:
                        best_task_id = node.task_id
                    else:
                        node.best_child = best_task_id
                        best_task_id = node.task_id
                    level_count += 1
            node = node.parent
        return

    def compare(self):

        mcts_cost = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))
        optimal_cost = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))
        est_cost = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))
        edt_cost = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))
        optimal_dropped = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))
        est_dropped = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))
        edt_dropped = np.zeros((self.args["ValidationSize"] + self.args["TestSize"]))

        mcts_nodes = []
        optimal_nodes = []
        for i_iter in range(self.args["ValidationSize"] + self.args["TestSize"]):
            # if (i_iter + 1) % 1 == 0:
            gc.collect()
            print('    Problem Number: ', i_iter + 1)
            

            # Produce problem
            # Task start times sorted in ascending order - Uniform(0, 98)
            start_time = np.sort(np.random.uniform(0, self.args["Total_time"] - 12, (1, self.args["NumTasks"])))
            # How long a task occupies the timeline - Uniform(2, 12)
            task_length = np.random.uniform(2, 15, (1, self.args["NumTasks"]))
            # Cost of delaying a task - Uniform(1, 5)
            tardiness_cost = np.random.uniform(1, 10, (1, self.args["NumTasks"]))
            # The last moment each task can be scheduled. After this time the task is dropped - Uniform(2, 12)
            drop_time = start_time + np.random.uniform(2, 12, (1, self.args["NumTasks"]))
            # Cost of dropping a task - Uniform(100, 500)
            drop_cost = np.random.uniform(100, 500, (1, self.args["NumTasks"]))

            problem_input = np.concatenate((start_time, task_length, tardiness_cost, drop_time, drop_cost,
                                            np.zeros((1, self.args["NumTasks"])), np.zeros((1, self.args["NumTasks"])),
                                            np.zeros((1, self.args["NumTasks"]))), axis=0)
            problem_input = problem_input.reshape((-1, self.args["NumFeatures"], self.args["NumTasks"], 1))\
                .astype(np.float64)

            problem_copy = copy.deepcopy(problem_input)

            # EST
            problem_copy = copy.deepcopy(problem_input)
            est_cost[i_iter], est_dropped[i_iter] = self.est_calc(problem_copy)

            problem_copy = copy.deepcopy(problem_input)

            # EDT
            edt_cost[i_iter], edt_dropped[i_iter] = self.edt_calc(problem_copy)


            # # MCTS
            # root = Node(self.args, problem_input)
            # self.search(root, FindOptimal = False)
            # mcts_nodes.append(root.problem_input)
            # mcts_cost[i_iter] = root.best_cost

            #Optimal
            root = Node(self.args, problem_input)
            self.search(root, FindOptimal = True)
            optimal_nodes.append(root.problem_input)
            optimal_cost[i_iter] = root.best_cost
            optimal_dropped[i_iter] = len(root.dropped_id_order_best)
           
        optimal_avg_cost = np.mean(optimal_cost)
        est_avg_cost = np.mean(est_cost)
        edt_avg_cost = np.mean(edt_cost)
                         
        optimal_avg_dropped = 100 * np.mean(optimal_dropped) / self.args["NumTasks"]
        print(optimal_avg_dropped)
        est_avg_dropped = 100 * np.mean(est_dropped) / self.args["NumTasks"]
        edt_avg_dropped = 100 * np.mean(edt_dropped) / self.args["NumTasks"]


        return optimal_avg_cost, est_avg_cost, edt_avg_cost, optimal_nodes, optimal_avg_dropped, est_avg_dropped, edt_avg_dropped
        # return mcts_cost, optimal_cost, est_cost, edt_cost, mcts_nodes, optimal_nodes


        

    def est_calc(self, problem_input):
        last_scheduled = 0  # counter for last scheduled task
        for i in range(self.args["NumTasks"]):
            # print(i)
            if i == 0:
                problem_input[0, 5, i, 0] = problem_input[0, 0, i, 0]  # execution time first task
                problem_input[0, 6, i, 0] = 1  # first task scheduled
                last_scheduled = i
            else:
                exec_time = max((problem_input[0, 5, last_scheduled, 0] +
                                 problem_input[0, 1, last_scheduled, 0]), problem_input[0, 0, i, 0])
                if exec_time <= self.args["Total_time"] and exec_time <= problem_input[0, 3, i, 0]:  # drop condition
                    problem_input[0, 5, i, 0] = exec_time  # task not dropped
                    problem_input[0, 6, i, 0] = 1  # i'th task scheduled
                    last_scheduled = i
                else:
                    problem_input[0, 7, i, 0] = 1  # i'th task dropped

        delays = problem_input[0, 5, :, 0] - problem_input[0, 0, :, 0]  # calculate all delays
        delays_masked = np.multiply(problem_input[0, 6, :, 0], delays)  # mask delays
        delays_cost = np.sum(np.multiply(problem_input[0, 2, :, 0], delays_masked))  # mask delays

        drops_cost = np.sum(np.multiply(problem_input[0, 7, :, 0], problem_input[0, 4, :, 0]))  # drop cost
        total_cost = delays_cost + drops_cost

        est_dropped = (np.sum(problem_input[0, 7, :, 0])) 

        return total_cost, est_dropped

    def edt_calc(self, problem_input):
        sorted_index = np.argsort(problem_input[0, 3, :, 0])
        last_scheduled = 0  # counter for last scheduled task
        for i in range(self.args["NumTasks"]):
            # print(i)
            if i == 0:
                problem_input[0, 5, sorted_index[i], 0] = problem_input[0, 0, sorted_index[i], 0]  # execution time first task
                problem_input[0, 6, sorted_index[i], 0] = 1  # first task scheduled
                last_scheduled = sorted_index[i]
            else:
                exec_time = max((problem_input[0, 5, last_scheduled, 0] +
                                 problem_input[0, 1, last_scheduled, 0]), problem_input[0, 0, sorted_index[i], 0])
                if exec_time <= self.args["Total_time"] and exec_time <= problem_input[0, 3, sorted_index[i], 0]:  # drop condition
                    problem_input[0, 5, sorted_index[i], 0] = exec_time  # task not dropped
                    problem_input[0, 6, sorted_index[i], 0] = 1  # i'th task scheduled
                    last_scheduled = sorted_index[i]
                else:
                    problem_input[0, 7, sorted_index[i], 0] = 1  # i'th task dropped

        delays = problem_input[0, 5, :, 0] - problem_input[0, 0, :, 0]  # calculate all delays
        delays_masked = np.multiply(problem_input[0, 6, :, 0], delays)  # mask delays
        delays_cost = np.sum(np.multiply(problem_input[0, 2, :, 0], delays_masked))  # mask delays

        drops_cost = np.sum(np.multiply(problem_input[0, 7, :, 0], problem_input[0, 4, :, 0]))  # drop cost
        total_cost = delays_cost + drops_cost

        edt_dropped = (np.sum(problem_input[0, 7, :, 0])) 

        return total_cost, edt_dropped
