import numpy as np
import unittest
from itertools import permutations

def dcg_array_tf(values, indices):
    l = np.zeros(values.shape)
    for i in range (len(values)):
        for j in range (len(values[0])):
            l[i][j] = values[i][indices[i][j]]
    return l

def dcg(values, indices = None):
    if indices is not None:
        values = dcg_array_tf(values, indices)
    i = np.log(1. + np.arange(1,len(values)+1))
    l = values
    
    return np.sum(l/i, axis = 1)
    
def create_rev_arr(values):
    l = np.zeros(values.shape)
    for i in range (len(values)):
        for j in range (len(values[0])):
            l[i][len(l[0])-j-1] = values[i][j]

    return l

class TestAdd(unittest.TestCase):
 
    def test_mondays_example(self):
        """
        Test that the addition of two integers returns the correct total
        """

        self.assertGreater(dcg(np.array([[1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0]]))[0],
                                                        dcg(create_rev_arr(np.array([[1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0]])))[0])
        
        self.assertGreaterEqual(dcg(np.array([[1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0]]))[0], 
                                
                                dcg(np.array([[0.0, 1, 0.6, 0.3], 
                                                        [0.0, 1, 0.6, 0.3], 
                                                        [0.0, 1, 0.6, 0.3], 
                                                        [0.0, 1, 0.6, 0.3]]), 
                                np.array([[2, 1, 3, 0], [2, 3, 1, 0], [1, 3, 2, 0], [2, 3, 1, 0]]))[0])

        '''print(dcg(np.array([[1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0]])))

        print(dcg(np.array([[0.0, 1, 0.6, 0.3], 
                                                        [0.0, 1, 0.6, 0.3], 
                                                        [0.0, 1, 0.6, 0.3], 
                                                        [0.0, 1, 0.6, 0.3]]), 
                                np.array([[2, 1, 3, 0], [2, 3, 1, 0], [1, 2, 3, 0], [1, 2, 0, 3]])))'''

        print(dcg(np.array([[1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0]]))[0],
                                                        dcg(create_rev_arr(np.array([[1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0], 
                                                        [1, 0.6, 0.3, 0.0]])))[0])

        new_list = []
        for k in permutations([1.0, 0.6, 0.3, 0.0]):
            new_list.append(list(k))
        
        for i in range (0,len(new_list),4):
            print(dcg(np.array(new_list[i : i + 4])))
 
if __name__ == '__main__':
    unittest.main()