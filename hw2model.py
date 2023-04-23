import numpy as np
import cupy as cp
def model():
    """
    Main method for my HW2. First, it loads the data, then it finds the SVD of the matrix M of rank r. 
    Loop method is the call to start the algorithm, then the results Afinal and Wfinal are the outputs of the algo. 
    Then, we take the negative entries to be 0 (to make it nonnegative), and use those as our final A and W. 

    """
    M,m,r = loadDataFile()
    #this code is to check if the premade A and W work properly. 
    checkPremadeAW(M,r,m)
    A0, W0 = svdMethod(M, r)
    iterationsBig = 1000
    iterationsSmall = 100
    epsilon = .001
    ac = 30
    wc = 30
    Afinal,Wfinal = loopMethod(A0, W0, M,iterationsBig, iterationsSmall, epsilon, wc, ac)
    Apos = Afinal>=0
    Wpos = Wfinal>=0
    A = Apos*Afinal
    W = Wpos*Wfinal
    loss = objectiveLoss(A,M, W)
    print("final rounded loss: ",loss)
    #printAWToFile(A,W)
    return

def loopMethod(A0, W0, M, iterationsBig, iterationsSmall, epsilon, wc, ac):
    """
    Method containing the optimization algorithm for finding the optimal A and W. 
    Uses alternating minmization. For a fixed A, find the W which minimizes the loss. 
    Then, for that new W, find an A that minimizes the loss. Repeat, and it will converge to an optimal 
    solutiion. 
    A0 - SVD calculated initial A. 
    W0 - SVD calculated initial W. 
    M - the matrix we want to factorize. 
    iterationsBig - number of iterations of alternating minmization. 
    iterationsSmall - number of iterations for gradient descent. 
    epsilon - for gradient descent. 
    wc - regularization coefficient for W
    ac - regularization coefficient for A
    """
    Aprev = A0
    Anew = A0
    Wprev = W0
    Wnew = W0

    for i in range(0, iterationsBig):
        Wnew = gradDescent(Wprev, Aprev, M, Wgrad, iterationsSmall, epsilon, wc)
        Anew = gradDescent(Aprev, Wnew, M, Agrad, iterationsSmall, epsilon, ac)
        print("Iteration: {}, loss: {}, roundedLoss: {}\n".format(i, objectiveLoss(Anew,M,Wnew), posLoss(Anew, M, Wnew)))
        Wprev = Wnew
        Aprev = Anew
    return Anew, Wnew
def gradDescent(X,Y, M, gradientFunction, numIterations, epsilon, xc):
    """
    This method represents a local optimization of A or W with the other value fixed. We do alternating minimization, and gradient 
    descent is how we minimize one of the two. 
    X - value we're optimizing (A or W)
    Y - the fixed value we're not optimizing (W or A).
    gradientFunction - the function Xgrad (either Agrad or Wgrad), 
    representing the gradient of the loss with respect to that parameter. 
    numIterations - how many steps of gradient descent. 
    epsilon - the parameter for GD. 
    xc - the regularization coefficient for X, either wc or ac. 
    """
    Xold = X
    Xnew = X
    for j in range(0, numIterations):
        Xnew = Xold -epsilon*gradientFunction(Xold,Y,M, xc)
        Xold = Xnew
    return Xnew
def Agrad(A, W, M, ac):
    """
    Helper method to take the gradient w.r.t A. 
    This is passed in as an input to the grad descent method if it's called on A. 
    Note that ac is the regularization coefficient for the negative value correction on A. 
    Calculates the gradient of the primary objective, the regularization, and adds them. 
    """
    basicGrad = gradA(A, M, W)
    negValue = gradANeg(A, ac)
    return basicGrad + negValue
def Wgrad(W,A,M,wc):
    """
    Helper method to take the gradient w.r.t W
    This is passed in as input to the gradDescent method if it's called on W.
    Note that wc is the regularization coefficient for the negative value correction on W. 
    Calculates the gradient of the primary objective, the regularization, and adds them.  
    """
    basicGrad = gradW(A,M,W)
    negValue = gradWNeg(W, wc)
    return basicGrad+negValue
def objectiveLoss(A,M,W):
    """
    The main objective loss function, given in the handout as L = ||M-AW||^2_F, 
    frobenius norm of the difference squared. This is what we take the gradient of 
    for gradient descent. 
    """
    return cp.square(cp.linalg.norm(M-A@W))
def posLoss(A, M, W):
    """
    Calculates the loss on the matrices rounded to be nonnegative. Sees how well it'll do when we round at the end. 
    """
    Apos = A>=0
    Wpos = W>=0
    Aadj = Apos*A
    Wadj = Wpos*W
    return objectiveLoss(Aadj, M, Wadj)
def gradA(A, M, W):
    """
    Gradient of the objective w.r.t A. 
    """
    difference = M-A@W
    return -2*difference@W.T
def gradW(A,M,W):
    """
    Gradient of the objective w.r.t W. 
    """
    return -2*A.T@(M-A@W)
def gradANeg(A, ac):
    """
    The gradient of the regularization term for A. 
    """
    Aneg = A<0
    return 2*ac* A*Aneg
def gradWNeg(W, wc):
    """
    The gradient of the regularization term for W. 
    """
    Wneg = W<0
    return 2*wc*W*Wneg
    
def nonnegativeLoss(A, W, ac, wc):
    """
    This is the loss term from A and W's negative entries. 
    """
    Aneg = A<0
    Wneg = W<0
    Astar = A*Aneg
    Wstar = W*Wneg
    
    Adist = cp.square(cp.linalg.norm(Astar))
    Wdist = cp.square(cp.linalg.norm(Wstar))
    return ac*Adist + wc*Wdist
def loadDataFile():
    """
    Loads M, r, m from the word_word_correlation file. 
    Creates a cupy/numpy array with them. 
    """
    fileName = "word_word_correlation"
    with open(fileName) as f:
        firstLineValues = f.readline().split()
        assert(len(firstLineValues)==2)
        m = int(firstLineValues[0])
        r = int(firstLineValues[1])
        #next m lines are m size and are the matrix. 
        listArrayRows = []
        for i in range(0, m):
            listArrayRows.append([float(value) for value in f.readline().split()])
        newArray = cp.row_stack(listArrayRows)
        assert(newArray.shape == (m, m))
        assert(cp.all(newArray == newArray.T))
        return newArray, m, r

def loadAW(m, r):
    """
    Loads the precalculated A and W from the nmf_ans.txt file. 
    This is primarily to check if they were stored correctly. 
    """
    fileName = "nmf_ans.txt"
    with open(fileName) as f:
        lines = f.readlines()
        listArrayRows = []
        for line in lines:
            listArrayRows.append([float(value) for value in line.split()])
        Alines = listArrayRows[0:m]
        Wlines = listArrayRows[m:]
        A = cp.row_stack(Alines)
        W = cp.row_stack(Wlines)
        assert(A.shape == (m, r))
        assert(W.shape == (r, m))
    return A, W
def printAWToFile(A, W):
    """
    Prints our generated A and W to the file nmf_ans.txt. 
    First m lines are A, with each line having r values. 
    Next r lines are W, with each line having m values. 
    """
    fileName = "nmf_ans.txt"
    m = A.shape[0]
    r = A.shape[1]
    with open(fileName, "w") as f:
        for i in range(m):
            Arow = A[i, :]
            ArowList = [str(value) for value in Arow]
            Astring = " ".join(ArowList)
            f.write(Astring + "\n")
        for j in range(r):
            Wrow = W[j,:]
            WrowList =  [str(value) for value in Wrow]
            Wstring = " ".join(WrowList)
            f.write(Wstring + "\n")
    return 
def svdMethod(M, r):
    """
    Calculate the svd of M, taking the top r eigenvectors. 
    This is used to initialize the gradient descent algorithm. 
    However, we set the negative values to be 0, to start it off right. 
    """
    u,s,v = np.linalg.svd(M)
    #U and V are mxm, so we cut them off. 
    Ucut = u[:, 0:r]
    Vcut = v[0:r, :]
    #this just gives the singular values, we put them into a diagonal matrix. 
    scut = np.diag(s[0:r])
    
    #Split the scut in two to get a factorization of two matrices. 
    Ai = Ucut@np.sqrt(scut)
    Wi = np.sqrt(scut)@Vcut
    loss = objectiveLoss(Ai, M, Wi)
    print("svd loss with negative: ", loss)
    #take the negative terms of the matrix to be 0. 
    Apos = Ai>=0
    A = Ai*Apos
    
    Wpos = Wi>=0
    W = Wi*Wpos
    posSvdLoss = objectiveLoss(A,M,W)
    print("initial svd loss: ", posSvdLoss)
    return A,W
def checkPremadeAW(M,r, m):
    """
    Checks if the A and W in nmf_ans are stored properly, and that they 
    get good results. 
    """
    A,W = loadAW(m,r)
    assert(cp.all(A>=0))
    assert(cp.all(W>=0))
    lossPrevious = objectiveLoss(A,M,W)
    print("Loss from file: ", lossPrevious)
    print("r value: ", r)
    print("m value: ", m)
    return
model()