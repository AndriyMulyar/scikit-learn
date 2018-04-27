from abc import ABC, abstractmethod
import numpy as np, math
from ._meta import ObliqueSplitRecord
"""
A collection of splitting criteria for decision trees
"""

"""
This method will need optimization. It contributes most the the constant term in induction complexity.
"""
def get_class_counts(instances, left_boundary, right_boundary, y):
    """

    :param instances: an array of references to instances
    :param left_boundary: the left boundary in instances array
    :param right_boundary: the right boundary in instances array
    :return: an array of length unique_classes holding class counts for each class
    """

    classes = []

    #find number of unique classes in this subset of instances
    #store in classes[]
    for i in range(left_boundary, right_boundary + 1):
        class_label = y[instances[i]]
        if not class_label in classes:
            classes.append(class_label)

    class_counts = np.zeros(len(classes))

    #calculate class counts and store in class_counts
    for i in range(left_boundary, right_boundary+1):

        for class_index in range(len(classes)):
            if y[instances[i]] == classes[class_index]:
                class_counts[class_index]+=1

        """
        Instead can do the slightly faster
        class_counts[y[instances[i]]] += 1
        but classes must be ordered integers starting at 0.
        Implemented code is more general.
        """

    return class_counts


class Criterion(ABC):

    def __init__(self, y):
        self.y = y

    @abstractmethod
    def calculate_impurity(self, instances, left_boundary, right_boundary):
        """
        Calculates the impurity of a given ObliqueSplitRecord
        :param split_record:
        :return: Sets fields split_record.left_impurity and split_record.right_impurity to appropriate values
        """
        if(left_boundary < 0 or  right_boundary>= len(self.y)):
            raise IndexError("Invalid left and right boundaries: %i %i" % (left_boundary, right_boundary))



class Gini(Criterion):

    def __init__(self, y):
        super().__init__(y)

    def calculate_impurity(self, instances, left_boundary, right_boundary):
        """

        Gini impurity is 1 - (sum of squares of each class probability over all classes)

        :param instances: instances array
        :param left_boundary: index in instances of the left_boundary_index of the split
        :param right_boundary: index in instances of the right_boundary_index of possible split
        :return:
        """
        num_instances = 1+right_boundary-left_boundary

        class_counts = get_class_counts(instances, left_boundary, right_boundary, self.y)

        gini = 0

        for c in range(len(class_counts)): #for each class

            prob_c = class_counts[c]/num_instances #calculate probability of class in subset
            gini += prob_c*prob_c

        return 1-gini #subtract from maximum value to turn into impurity metric




class Hellinger(Criterion):

    def __init__(self, y):
        super().__init__(y)

    def calculate_impurity(self, instances, left_boundary, right_boundary):
        """
        A skew insensitive impurity metric

        Definition can be found here: https://en.wikipedia.org/wiki/Hellinger_distance#Discrete_distributions
        :param split_record:
        :return:
        """

        num_instances = 1 + right_boundary - left_boundary

        class_counts = get_class_counts(instances, left_boundary, right_boundary, self.y)

        hellinger = 0

        for c1 in range(len(class_counts)): #for each class
            for c2 in range(c1):
                prob_c1 = class_counts[c1]/num_instances
                prob_c2 = class_counts[c2]/num_instances

                pair_wise_distance = math.sqrt(prob_c1) - math.sqrt(prob_c2)
                hellinger += pair_wise_distance*pair_wise_distance

        return 1 - math.sqrt(hellinger/2)