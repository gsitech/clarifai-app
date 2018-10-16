'''Functions to import data for training and testing'''

#TODO
# Modify the file paths of these functions once you have 
# enough data to train and test separately

def train_input_fn():
    """Training data supplier.
    """

    with open('data/embeddings.txt', 'r') as emb, open('data/labels.txt', 'r') as lab:
        x=[]
        y=[]
        for line in emb:
            x_=[]
            for i in line.split(","):
                x_.append(float(i))
            x.append(x_)
            
        for line in lab:
            y.append([int(line)])

    return x, y


def test_input_fn():
    """Test input function.
    """

    with open('data/embeddings.txt', 'r') as emb, open('data/labels.txt', 'r') as lab:
        x=[]
        y=[]
        for line in emb:
            x_=[]
            for i in line.split(","):
                x_.append(float(i))
            x.append(x_)
            
        for line in lab:
            y.append([int(line)])

    return x, y

def get_labels_as_strings():
    """Gets the string values of the labels
    """
    y=[]
    with open('data/labels_strings.txt', 'r') as lab:
        for line in lab:
            y.append(line[0:-1])

    return y

