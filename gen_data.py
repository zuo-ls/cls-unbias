import numpy as np
from core.util import seed_everything


def generate_gaussian_data(n, data_requirements):
    """Generate Gaussian data with specific requirements.
    
    Args:
        n (int): Target number of samples
        data_requirements (list): List of tuples specifying probabilities for each class
        
    Returns:
        np.ndarray: Generated data matrix
    """
    max_attempts = 100  # Prevent infinite loops
    attempt = 0
    
    while attempt < max_attempts:
        data_n = [
            (n * np.array(each_prob)).astype(int)
            for each_prob in data_requirements
        ]
        list_n = np.array([sum(data_n[i]) for i in range(len(data_n))])

        if (list_n[0] == n) and np.var(list_n) == 0.:
            break
        else:
            n += 1
        attempt += 1
    
    if attempt >= max_attempts:
        raise ValueError(f"Could not satisfy data requirements after {max_attempts} attempts")

    data = [
        np.concatenate([
            np.random.randn(each_n[0]),  # Already has randomness from normal distribution
            np.random.randn(each_n[1]) + 3
        ])
        for each_n in data_n
    ]

    data = np.stack(data, axis=1)
    np.random.shuffle(data)
    return data


def get_train_two_feats_gau(N,pos_prob, do_spurious=True, seed=0):
    seed_everything(seed)
    if do_spurious:
        x1_req = [(0, 1.),(0.,1.0),]# 0.1
        x2_req = [(1, 0.),(0.,1.0),]# 0.9

        x3_req = [(1, 0.),(0.5,0.5),] # 0.1
        x4_req = [(0, 1.),(0.5,0.5),] # 0.9
    else:
        x1_req = [(0, 1.),(0.5,0.5),] # 0.1
        x2_req = [(1, 0.),(0.5,0.5),] # 0.9

        # cls = 1
        x3_req = [(1, 0.),(0.5,0.5),] # 0.1
        x4_req = [(0, 1.),(0.5,0.5),] # 0.9

    neg_prob = 1 - pos_prob

    n1 = int(N * neg_prob * 0.1)
    n2 = int(N * neg_prob * 0.9)
    n3 = int(N * pos_prob * 0.1)
    n4 = int(N * pos_prob * 0.9)

    print(n1, n2, n3, n4)

    data_list = [
        generate_gaussian_data(n1, x1_req),
        generate_gaussian_data(n2, x2_req),
        generate_gaussian_data(n3, x3_req),
        generate_gaussian_data(n4, x4_req)
    ]

    n_list = [len(data) for data in data_list]
    train_x = np.concatenate(data_list)
    train_y = np.concatenate([
        np.zeros(n_list[0] + n_list[1]),
        np.ones(n_list[2] + n_list[3])
    ])
    return train_x,train_y

def get_test_two_feats_gau(N,pos_prob, do_rand=False, seed=0):
    seed_everything(seed)
    if do_rand:
        # test cls = 0
        x_5_req = [(1, 0.),(0.5,0.5),] # 0.9
        x_6_req = [(0, 1.),(0.5,0.5),] # 0.1

        # test cls = 1
        x_7_req = [(0, 1.),(0.5,0.5),] # 0.9
        x_8_req = [(1, 0.),(0.5,0.5),] # 0.1
    else:
        # test cls = 0
        x_5_req = [(1, 0.),(0.,1.0),] # 0.9
        x_6_req = [(0, 1.),(0.,1.0),] # 0.1

        # test cls = 1
        x_7_req = [(0, 1.),(0.,1.0),] # 0.9
        x_8_req = [(1, 0.),(0.,1.0),] # 0.1

    neg_prob = 1 - pos_prob

    n5 = int(N * neg_prob * 0.9)
    n6 = int(N * neg_prob * 0.1)
    n7 = int(N * pos_prob * 0.9)
    n8 = int(N * pos_prob * 0.1)

    data_list = [
        generate_gaussian_data(n5, x_5_req),
        generate_gaussian_data(n6, x_6_req),
        generate_gaussian_data(n7, x_7_req),
        generate_gaussian_data(n8, x_8_req),
    ]

    n_list = [len(data) for data in data_list]
    test_x = np.concatenate(data_list)
    test_y = np.concatenate([
        np.zeros(n_list[0] + n_list[1]),
        np.ones(n_list[2] + n_list[3])
    ])
    return test_x,test_y