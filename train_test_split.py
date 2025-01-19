def train_test_split(X, y, test_size, shuffle = True):
    if len(X) != len(y):
        raise ValueError(f"Features (len={len(X)}) and labels (len={len(y)}) must have the same length!")
    
    if test_size < 0 or test_size > 1:
        raise ValueError(f"Invalid test size {test_size}! Choose a value between 0 and 1.")
    
    data = list(zip(X,y))

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    X_train = list(X_train)
    X_test = list(X_test)
    y_train = list(y_train)
    y_test = list(y_test)

    return X_train, X_test, y_train, y_test