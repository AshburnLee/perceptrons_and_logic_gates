import numpy as np
import matplotlib.pyplot as plt

"""
Class implementation of a perceptron 
"""
class Perceptron(object):
   
    def __init__(self, x, y):

        self.numFeature = x.shape[1]  # instance dimention 
        self.bias = 0.0   # init bias 
        self.weights = np.zeros(x.shape[1])  # init weights
        self.inputs = x  # dataset array
        self.label = y  # labels array or list 
        self.errorList = []   # store number of errors

    def output(self, x): 
        """ Return the output (0 or 1) from the perceptron"""
        tmp = np.dot(self.weights, x) + self.bias
        if tmp > 0:
            return 1
        else:
            return 0

    def learn4setosa(self, eta=0.01, showProcess=True):
        """ learning process, stop learning untill the number of errors is 0."""
        self.bias = np.random.normal()
        self.weights = np.random.randn(self.numFeature)
        numError = 100
        numIter = 0


        while numError != 0:  # setosa 
            numError = 0
            if showProcess:  # show process
                print ("Bias: {:.3f}".format(self.bias))
                print ("Weights:", ", ".join("{:.3f}".format(wt) for wt in self.weights))

            for x,y in zip(self.inputs, self.label):
                error = y - self.output(x)    # scale - scale
                if error:   # if error != 0, update
                    numError += 1
                    """update the member attributes, the final learned weights and bias
                    will be stored in self.bias and self.weights"""
                    self.bias += eta*error 
                    self.weights += eta*error*x
                # print(error)  # show errors in 150 instances in i iteration

            numIter += 1
            self.errorList.append(numError)
            if showProcess:
                print ("Number of iteration: ", numIter)
                print ("Number of errors:", numError, "\n")

    def learn4virginica(self, eta=0.005,showProcess=True):
        """ learning process, stop learning untill the number of errors is 0."""
        self.bias = np.random.normal()
        self.weights = np.random.randn(self.numFeature)
        numError = 100
        numIter = 0

        while (numIter < 1000 or numError > 2):  # viginica run numError = 0 as a prior, and replace the min with 0
            numError = 0
            if showProcess:
                print ("Bias: {:.3f}".format(self.bias))
                print ("Weights:", ", ".join("{:.3f}".format(wt) for wt in self.weights))

            for x,y in zip(self.inputs, self.label):
                error = y - self.output(x)    # scale - scale
                if error:   # if error != 0, update
                    numError += 1
                    """update the member attributes, the final learned weights and bias
                    will be stored in self.bias and self.weights"""
                    self.bias += eta*error 
                    self.weights += eta*error*x
                #print(error)  # show errors in 150 instances in i iteration
            numIter += 1
            self.errorList.append(numError)
            if showProcess:
                print ("Number of iteration: ", numIter)
                print ("Number of errors:", numError, "\n")

    def plotError(self):
        """Thsi function is used for plotting the number of errors"""
        yAxis = np.linspace(1, len(self.errorList), len(self.errorList))
        plt.plot(yAxis, self.errorList,'o-')
        plt.xlabel("iteration")
        plt.ylabel("number of errors")
        plt.show()


        """after learning, call the following fuctions"""
    # def test(self, x, y):
    #     """pass single instance and the corresponding label y"""
    #     tmp = self.output(x)
    #     return "match" if (y-tmp)==0 else "mismatch"

    # def predict(self, x):
    #     """pass a instance, predict the label"""
    #     tmp = self.output(x)
    #     return "mango" if tmp==0 else "non-mango"

    def outAsIn(self, x):
        frstLayer = []
        for _ in x:
            # print(self.output(_))
            frstLayer.append(self.output(_))
        return frstLayer   # first output for next layer
