# Implement square root (hihi)
def sqrt(num: float) -> float:
    return num**0.5

# Implement euclidean distance of two points (no matter the dimension)
def euclidean_distance(p1, p2) -> float:
    # if one point is null, distance = infinite
    if not p1 or not p2:
        return float('inf')
    
    elif len(p1) != len(p2):
        raise ValueError("Points must have same dimension")

    d = 0
    for i in range(len(p1)):
        d += (p1[i] - p2[i])**2
    return sqrt(d)

# Implement manhatten (taxicab) distance of two points
def manhatten_distance(p1, p2) -> float:
    # if one point is null, distance = infinite
    if not p1 or not p2:
        return float('inf')
    
    elif len(p1) != len(p2):
        raise ValueError("Points must have same dimension")
    
    d = 0
    for i in range(len(p1)):
        d += abs(p1[i] - p2[i])
    return d

# Minkowski distance: if p = 1 -> manhatten distance, p = 2 -> euclidean distance, p = 3 -> ... (infinite number of distance metrics)
def minkowski_distance(p1, p2, p) -> float:
    # if one point is null, distance = infinite
    if not p1 or not p2:
        return float('inf')
    
    elif len(p1) != len(p2):
        raise ValueError("Points must have same dimension")
    
    d = 0
    for i in range(len(p1)):
        d += abs(p1[i] - p2[i])**p
    return d**(1/p)

# Returns 1 if two objects are equal, 0 if not. Used for classification later on
def kronecker_delta(v1, v2):
    if v1 == v2:
        return 1
    else:
        return 0