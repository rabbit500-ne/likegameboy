def pthj(A, B):
    if A[-1] == "/":
        return A + B
    else:
        return A + "/" + B