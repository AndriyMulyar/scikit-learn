# cython: profile=True
from abc import ABC, abstractmethod
import numpy as np
cimport numpy as np
from ._meta import DecisionNode, ObliqueSplitRecord, hyperplane_compare

from ._criterion import Criterion, Gini, Hellinger, get_class_counts
"""
Bi-partitions the instances in a given DecisionNode based on a particular splitting algorithm.
"""





class ObliqueSplitter(ABC):
    """
    An interface defining a Splitter
    """

    @abstractmethod
    def __init__(self, X, y, criterion, random_state):
        self.criterion = criterion
        self.X = X
        self.y = y
        if random_state is None:
            self.random_state = np.random.RandomState(1)
        else:
            self.random_state = random_state


    @abstractmethod
    def split(self, node, instances):
        """
        Splits a given DecisionNode using a defined splitting algorithm
        :return: An ObliqueSplitRecord that defines the split of given DecisionNode
        """
        #pre-flight checks
        if not isinstance(node, DecisionNode):
            raise TypeError("Split can only be called on a DecisionNode")

        if node.instances_in_node is None:
            raise ValueError("instances_in_node not set")
        if node.left_boundary_index is None:
            raise ValueError("left_boundary_index not set")
        if node.right_boundary_index is None:
            raise ValueError("right_boundary_index not set")


    def sort_along_hyperplane(self, instances, hyperplane, left_boundary, right_boundary):
        """
        Splits instances between left and right boundary according to whether they are above or below hyperplane
        To simplify understanding, interpret below hyperplane as belonging to left child and above hyperplane
        as belonging to right child.
        :param instances:
        :param hyperplane:
        :param left_boundary:
        :param right_boundary:
        :return: split_index defining the partition
        """

        cdef int current_left
        cdef int j

        current_left = left_boundary

        while current_left <= right_boundary:

            while current_left <= right_boundary and hyperplane_compare(self.X[instances[current_left]], hyperplane) <= 0:
                current_left+= 1
            j = current_left+1
            while j <= right_boundary and hyperplane_compare(self.X[instances[j]], hyperplane) >= 0:
                j+=1
            if j > right_boundary:
                return current_left
            #current_left is the index of an instance above the hyperplane
            #j is the index of an element below the hyperplane

            instances[current_left], instances[j] = instances[j], instances[current_left]

        return current_left


    def sort_along_feature(self, instances, feature, left_boundary, right_boundary):
        """
        Sorts the array instances on the feature 'feature' between left_boundary and right_boundary.
        Implements quicksort, adds a complexity factor of theta of num_instances*log(num_instances) where
        num_instances = 1 + right_boundary - left_boundary

        Implementation modified from https://stackoverflow.com/questions/18262306/quicksort-with-python

        Naively chooses first element as pivot, could do better if needed.

        :param instances: an array of references to instances in X (rows)
        :param feature: the feature to sort on, 0 <= feature < dimensionality
        :param left_boundary: the left most index of instances to include in sorting
        :param right_boundary: the right most index of instances to include in sorting
        :return: instances is appropriately sorted along the given feature between the given indices
        """
        if left_boundary>=right_boundary:
            return
        pivot = self._partition(instances,feature,left_boundary,right_boundary)
        #print("pivot: %i" % pivot)
        #print("left: %i" % left_boundary)
        self.sort_along_feature(instances, feature, left_boundary, pivot-1)
        self.sort_along_feature(instances, feature, pivot+1, right_boundary)

    def _partition(self, instances, feature, begin, end):
        """

        :param instances:an array of references to instances in X (rows)
        :param feature: feature to sort on
        :param begin: beginning index to partition instances on
        :param end: end index to partition instances on
        :return:
        """
        pivot = begin
        for i in range(begin+1, end+1): #find appropriate location for new pivot while maintaining order property
            if self.X[instances[i]][feature] <= self.X[instances[begin]][feature]:
                pivot += 1
                instances[i], instances[pivot] = instances[pivot], instances[i]
        instances[pivot], instances[begin] = instances[begin], instances[pivot]

        return pivot

class AxisParallelSplitter(ObliqueSplitter):

    def __init__(self,X,y,criterion, random_state = None):
        super().__init__(X,y,criterion, random_state)



    def split(self, node, instances):
        """
        Finds the best axis parallel split on instances and stores it in node.split_record
        :param node: a DecisionNode
        :param instances: an array of reference indices to rows in X
        """

        cdef int f
        cdef int candidate_split_index


        super().split(node, instances)  #pre-flight checks
        dimension = self.X.shape[1]
        best_split = ObliqueSplitRecord(dimension)
        current_split = ObliqueSplitRecord(dimension)

        best_split.impurity_total = self.criterion.calculate_impurity(instances, node.left_boundary_index, node.right_boundary_index)
        current_split.impurity_total = best_split.impurity_total




        for f in range(dimension): #for each feature
            #print(f)
            """
            1. Sort instances along feature f
            2. Calculate impurity for each possible split along the sorted instances
            3. update best_split_record if goodness of split is greater in a new found split than previous max
            """
            current_split.hyperplane.fill(0) #reset hyperplane
            current_split.hyperplane[f] = 1 #specify which dimension the new potential best hyperplane resides in
            self.sort_along_feature(instances, f, node.left_boundary_index, node.right_boundary_index)

            #for each candidate split in our now sorted features, consider it.
            for candidate_split_index in range(node.left_boundary_index+1, node.right_boundary_index+1):
                # if self.y[instances[candidate_split_index]] == self.y[instances[candidate_split_index-1]]:
                #     continue

                current_split.impurity_left = self.criterion.calculate_impurity(instances,node.left_boundary_index,
                                                                                candidate_split_index)
                current_split.number_of_instances_left = candidate_split_index - node.left_boundary_index

                current_split.impurity_right = self.criterion.calculate_impurity(instances, candidate_split_index,
                                                                                 node.right_boundary_index)
                current_split.number_of_instances_right = 1 + node.right_boundary_index - candidate_split_index

                if current_split.get_goodness_of_split() >= best_split.get_goodness_of_split():
                    current_split.hyperplane[-1] = -1*(self.X[instances[candidate_split_index]][f] +
                                                       self.X[instances[candidate_split_index-1]][f])/2.0
                    best_split.impurity_left = current_split.impurity_left
                    best_split.impurity_right = current_split.impurity_right
                    best_split.number_of_instances_left = current_split.number_of_instances_left
                    best_split.number_of_instances_right = current_split.number_of_instances_right
                    best_split.hyperplane = np.copy(current_split.hyperplane)
                    best_split.split_index = candidate_split_index-1




        """
        best_split now holds the values for the best possible axis_parallel split on the given node.
        instances, however, is still sorted according to the last feature considered.
        re-sort instances according to the best splitting hyperplane.
        """
        self.sort_along_hyperplane(instances, best_split.hyperplane, node.left_boundary_index,
                                   node.right_boundary_index)
        #TODO print("Best parallel hyperplane: "+ str(best_split.hyperplane))
        node.split_record = best_split





class AxisParallelDynamicImpuritySplitter(ObliqueSplitter):

    def __init__(self,X,y,criterion, random_state = None, imbalance_ratio_threshold = 60):
        super().__init__(X,y,criterion, random_state)
        self.imbalance_ratio_threshold = imbalance_ratio_threshold



    def split(self, node, instances):
        """
        Finds the best axis parallel split on instances and stores it in node.split_record

        Determines impurity criterion to use when testing for potential splits by making use of the imabalance
        ratio at the current node.


        :param node: a DecisionNode
        :param instances: an array of reference indices to rows in X
        """

        cdef int f
        cdef int candidate_split_index


        super().split(node, instances)  #pre-flight checks

        class_counts = get_class_counts(instances, node.left_boundary_index, node.right_boundary_index, self.y)
        min_class_count = np.amin(class_counts)
        max_class_count = np.amax(class_counts)

        if max_class_count/min_class_count > self.imbalance_ratio_threshold: #use hellinger for all splits
            self.criterion = Hellinger(self.y)
            node.split_record.criterion_used = "hellinger"
        else:
            self.criterion = Gini(self.y)
            node.split_record.criterion_used = "gini"

        #print(self.criterion)


        dimension = self.X.shape[1]
        best_split = ObliqueSplitRecord(dimension)
        current_split = ObliqueSplitRecord(dimension)
        best_split.impurity_total = self.criterion.calculate_impurity(instances, node.left_boundary_index, node.right_boundary_index)
        current_split.impurity_total = best_split.impurity_total




        for f in range(dimension): #for each feature
            #print(f)
            """
            1. Sort instances along feature f
            2. Calculate impurity for each possible split along the sorted instances
            3. update best_split_record if goodness of split is greater in a new found split than previous max
            """
            current_split.hyperplane.fill(0) #reset hyperplane
            current_split.hyperplane[f] = 1 #specify which dimension the new potential best hyperplane resides in
            self.sort_along_feature(instances, f, node.left_boundary_index, node.right_boundary_index)

            #for each candidate split in our now sorted features, consider it.
            for candidate_split_index in range(node.left_boundary_index+1, node.right_boundary_index+1):
                # if self.y[instances[candidate_split_index]] == self.y[instances[candidate_split_index-1]]:
                #     continue

                current_split.impurity_left = self.criterion.calculate_impurity(instances,node.left_boundary_index,
                                                                                candidate_split_index)
                current_split.number_of_instances_left = candidate_split_index - node.left_boundary_index

                current_split.impurity_right = self.criterion.calculate_impurity(instances, candidate_split_index,
                                                                                 node.right_boundary_index)
                current_split.number_of_instances_right = 1 + node.right_boundary_index - candidate_split_index

                if current_split.get_goodness_of_split() >= best_split.get_goodness_of_split():
                    current_split.hyperplane[-1] = -1*(self.X[instances[candidate_split_index]][f] +
                                                       self.X[instances[candidate_split_index-1]][f])/2.0
                    best_split.impurity_left = current_split.impurity_left
                    best_split.impurity_right = current_split.impurity_right
                    best_split.number_of_instances_left = current_split.number_of_instances_left
                    best_split.number_of_instances_right = current_split.number_of_instances_right
                    best_split.hyperplane = np.copy(current_split.hyperplane)
                    best_split.split_index = candidate_split_index-1




        """
        best_split now holds the values for the best possible axis_parallel split on the given node.
        instances, however, is still sorted according to the last feature considered.
        re-sort instances according to the best splitting hyperplane.
        """
        self.sort_along_hyperplane(instances, best_split.hyperplane, node.left_boundary_index,
                                   node.right_boundary_index)
        #TODO print("Best parallel hyperplane: "+ str(best_split.hyperplane))
        node.split_record = best_split





class OC1Splitter(ObliqueSplitter):

    def __init__(self,X,y,criterion,random_state=None,restarts=20, random_pert = 5):
        super().__init__(X,y,criterion, random_state)

        self.restarts = restarts
        self.random_pert = random_pert

    def perturb(self, instances, current_split, f, left_boundary_index, right_boundary_index):
        """
        Perturbs current_split.hyperplane on dimension f for node between left_boundary_index and right_boundary_index
        :param instances:
        :param current_split: the current split record
        :param f: dimensions to perturbate
        :param left_boundary_index:
        :param right_boundary_index:
        :return: True if better split was found during
                current_split now contains a better hyperplane after perturbation on dimension f
        """
        cdef int i
        cdef double U_j
        cdef double a_f

        cdef np.ndarray[double, ndim=1] hyperplane
        cdef np.ndarray[double, ndim=1] U

        hyperplane = current_split.hyperplane
        U = np.zeros(right_boundary_index-left_boundary_index+1, dtype=np.double)

        for i in range(left_boundary_index, right_boundary_index + 1):
            if self.X[instances[i]][f] == 0:
                U[i-left_boundary_index] = 0
            else:
                U_j = (hyperplane[f] * self.X[instances[i]][f] - hyperplane_compare(self.X[instances[i]], hyperplane)) / \
                      self.X[instances[i]][f]
                U[i - left_boundary_index] = U_j

        #print(right_boundary_index-left_boundary_index)
        U = np.sort(U)

        # find best univariate split of U's
        temp_split = ObliqueSplitRecord(current_split.dimensions)
        temp_split.impurity_total = current_split.impurity_total
        temp_split.hyperplane = np.copy(current_split.hyperplane)

        better_split_found = False
        for i in range(1, U.shape[0]):

            a_f = (U[i] + U[i - 1]) / 2
            temp_split.hyperplane[f] = a_f
            candidate_split_index = self.sort_along_hyperplane(instances, temp_split.hyperplane, left_boundary_index,
                                                               right_boundary_index)

            temp_split.impurity_left = self.criterion.calculate_impurity(instances, left_boundary_index, candidate_split_index-1)
            temp_split.impurity_right = self.criterion.calculate_impurity(instances, candidate_split_index, right_boundary_index)
            temp_split.number_of_instances_left = candidate_split_index - left_boundary_index
            temp_split.number_of_instances_right = 1 + right_boundary_index - candidate_split_index

            if temp_split.get_goodness_of_split() > current_split.get_goodness_of_split():
                better_split_found = True
                current_split.impurity_left = temp_split.impurity_left
                current_split.impurity_right = temp_split.impurity_right
                current_split.number_of_instances_right = temp_split.number_of_instances_right
                current_split.number_of_instances_left = temp_split.number_of_instances_left
                current_split.hyperplane = np.copy(temp_split.hyperplane)
                current_split.split_index = candidate_split_index-1

        return better_split_found

    def add_random_hyperplane(self, hyperplane):
        for d in range(len(hyperplane)):
            hyperplane[d] += self.random_state.random_sample()*2 -1 #add random float between [-1, 1)


    def split(self, node, instances):
        """
        Implements OC1 to find the best oblique split on instances between node.left_boundary_index and
        node.right_boundary_index

        The initial hyperplane to perturbate is initialized to the best axis parallel hyperplane.
        This guarantees that OC1 will find a split atleast as good as the axis parallel splitting algorithm.

        http://legacydirs.umiacs.umd.edu/~salzberg/docs/murthy_thesis/node11.html

        Algorithm outline:

        Attempt self.iterations times:
            Generate random hyperplane except on initial iteration - set this one to best.


            for each feature:
                perturbate hyperplane by modifying the feature value

        Sets node.split_record to the best OC1 split between node.left_boundary_index and node.right_boundary_index
        :param node: A DecisionNode
        :param instances:
        :return: nothing, simply updates the fields as nodes as postcondition
        """

        super().split(node, instances)
        axis_parallel_splitter = AxisParallelSplitter(self.X, self.y, self.criterion, self.random_state)

        axis_parallel_splitter.split(node, instances)
        #node.split_record now contains the best axis parallel split

        best_split = node.split_record

        current_split = ObliqueSplitRecord(best_split.dimensions)


        #initialize current split to best axis parallel split
        current_split.hyperplane = np.copy(best_split.hyperplane)
        current_split.impurity_total = best_split.impurity_total
        #done copy
        for R in range(self.restarts+1):
            #TODO temp comment print(R)
            if R > 1: #first iteration is with best axis parallel splits as starting hyperplane, other iterations are with random plane
                random_plane = np.zeros(best_split.dimensions) #initialize to zero vector
                self.add_random_hyperplane(random_plane) #add random components
                current_split.hyperplane = random_plane
                self.sort_along_hyperplane(instances, current_split.hyperplane, node.left_boundary_index,
                                           node.right_boundary_index)
                current_split.impurity_total = self.criterion.calculate_impurity(instances, node.left_boundary_index,
                                           node.right_boundary_index)



            # perturbate each dimension of hyperplane in sequence
            def pert_seq():
                for d in range(node.split_record.dimensions):
                    #keep perturbating co-efficient until no changes can be made
                    better_split_exists = True
                    while better_split_exists:
                        better_split_exists = self.perturb(instances, current_split, d, node.left_boundary_index, node.right_boundary_index)

            pert_seq()

            # update best hyperplane if new one found is more pure
            if current_split.get_goodness_of_split() > best_split.get_goodness_of_split():
                best_split.hyperplane = np.copy(current_split.hyperplane)
                best_split.impurity_left = current_split.impurity_left
                best_split.number_of_instances_left = current_split.number_of_instances_left
                best_split.impurity_right = current_split.impurity_right
                best_split.number_of_instances_right = current_split.number_of_instances_right
                best_split.split_index = current_split.split_index
            #perturbate in random direction

            for j in range(self.random_pert):

                self.add_random_hyperplane(current_split.hyperplane)
                self.sort_along_hyperplane(instances, current_split.hyperplane, node.left_boundary_index,
                                           node.right_boundary_index)
                current_split.impurity_total = self.criterion.calculate_impurity(instances, node.left_boundary_index,
                                                                                 node.right_boundary_index)
                #random_direction = self.random_state.randint(0,best_split.dimensions)
                better_split_exists = self.perturb(instances, current_split, 0, node.left_boundary_index,
                                               node.right_boundary_index)
                if better_split_exists:
                    pert_seq()


            #update best hyperplane if new one found is more pure
            if current_split.get_goodness_of_split() > best_split.get_goodness_of_split():
                best_split.hyperplane = np.copy(current_split.hyperplane)
                best_split.impurity_left = current_split.impurity_left
                best_split.number_of_instances_left = current_split.number_of_instances_left
                best_split.impurity_right = current_split.impurity_right
                best_split.number_of_instances_right = current_split.number_of_instances_right
                best_split.split_index = current_split.split_index


        self.sort_along_hyperplane(instances, best_split.hyperplane, node.left_boundary_index,
                                   node.right_boundary_index)
        #print(current_split.hyperplane)
        node.split_record = best_split















