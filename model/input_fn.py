"""Create the input data pipeline using `tf.data`"""

#import model.mnist_dataset as mnist_dataset


def train_input_fn():
    """Training data supplier.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    with open('model/embeddings.txt', 'r') as emb, open('model/labels.txt', 'r') as lab:
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
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    print("Test data generation")

    with open('model/embeddings.txt', 'r') as emb, open('model/labels.txt', 'r') as lab:
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

