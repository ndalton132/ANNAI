import numpy as np
import random as rand
import math


#Returns 1D array, w/ tuple(normalizedArray, character)
def readFromFile(path):
    with open(path, 'r') as file:
        # Initialize an empty list to store characters
        characters_array = []

        # Read each line in the file
        for line in file:
            # Split each line into individual characters and store them in the characters_array
            characters = [char for char in line.strip()]
            characters_array.extend(characters)
        
        array_2d = [characters_array[i:i+63] for i in range(0, len(characters_array), 63)]
        
        normalized_array = [[1 if char == '#' else 0 for char in row] for row in array_2d]
        
        # Return the 2D array
        return normalized_array
    

        
            
def createData(numHidden):
        lettersArray = readFromFile("C:\\Users\\nickd\\Downloads\\HW3_Training.txt")
        
        initVal = forwProp.weightInit(63) * .01
                
        W1 = [[rand.uniform(0, .05) for i in range(63)] for j in range(numHidden)]
        B1 = [rand.uniform(0, .05) for i in range(numHidden)]
        
        W2 = [[rand.uniform(0,.05) for i in range(numHidden)] for j in range(7)]
        B2 = [rand.uniform(0,.05) for i in range(7)]
        
        return lettersArray, W1, W2, B1, B2
    

def main():
    #Number of nodes in the hidden
    #Returns the initial weight matrix and Bias matrix
    inputLayer, W1, W2, B1, B2 = createData(150)
 
    
    inputLayer = Layer(WeightMatrix=[],BiasMatrix=[],ZVals=inputLayer,AVals=[])

    hiddenLayer = Layer(WeightMatrix=W1,BiasMatrix=B1,ZVals=[],AVals=[])
    
    outputLayer = Layer(WeightMatrix=W2,BiasMatrix=B2,ZVals=[],AVals=[])
    
    correctValues = ["A","B","C","D","E","J","K","A","B","C","D","E","J","K","A","B","C","D","E","J","K"]
    
    
    index = 0
    #Sum all values from node to each middle layer node
    #Input Layer ----> hiddenLayer
    #Iterate over each example in the input Layer, contains each example
    #for letter in inputLayer:
    #initilized to input 0 for now
    #forwardPropogateInit(inputLayer, weightMatrix, bias, nextLayer, ZArray):
    
    #for i in range(0,len(inputLayer.ZVals) - 1):
    for i in range(0,len(inputLayer.ZVals)):
        
        forwardPropogateInit(inputLayer.ZVals[i], hiddenLayer.WeightMatrix, hiddenLayer.BiasMatrix, hiddenLayer.AVals, hiddenLayer.ZVals, 0)
        
        forwardPropogateLayer(hiddenLayer.ZVals, outputLayer.WeightMatrix, outputLayer.BiasMatrix, outputLayer.AVals, outputLayer.ZVals, 0)

        

        #Calculate sum of error between the output and the expected
        #cost(correctValueLust,outputLayer, The value that was expected)
        
        # format(hiddenLayer.WeightMatrix)
        # print("")
        # format(outputLayer.WeightMatrix)
        # print("\n")
        
        # format(hiddenLayer.ZVals)
        # print("")
        # format(outputLayer.ZVals)
        # print("\n")
        
        
        #cost(resultList, outputLayer, expectedCharacter)
        Cost = (sum(outputLayer.AVals) - 1)**2
        derivCost = 2*(sum(outputLayer.AVals) - 1) 
        
        print(outputLayer.AVals)
        #print("COST: ",Cost)
        
            #print(hiddenLayer.WeightMatrix)
        outputLayer.backWardPropogate(hiddenLayer,derivCost)
        hiddenLayer.backWardPropogate(inputLayer,derivCost)
        
        
        
        hiddenLayer.AVals = []
        hiddenLayer.ZVals = []
        
        outputLayer.AVals = []
        outputLayer.ZVals = []
        
        
        
    
    #Check correct Values
    for i in range(0,len(inputLayer.ZVals) - 1):

        forwardPropogateInit(inputLayer.ZVals[i], hiddenLayer.WeightMatrix, hiddenLayer.BiasMatrix, hiddenLayer.AVals, hiddenLayer.ZVals, 0)

        forwardPropogateLayer(hiddenLayer.ZVals, outputLayer.WeightMatrix, outputLayer.BiasMatrix, outputLayer.AVals, outputLayer.ZVals, 0)
        guess = max(outputLayer.AVals)
        print(outputLayer.AVals)
        print(outputLayer.AVals.index(guess))
        
        hiddenLayer.AVals = []
        hiddenLayer.ZVals = []
        
        outputLayer.AVals = []
        outputLayer.ZVals = []
    
    
    
def format(Array):
    for i in range(len(Array)):
        print(Array[i])
    


def forwardPropogateInit(inputLayer, weightMatrix, bias, nextLayer, ZArray, debug):
    
    #This function takes the sum of input nodesXthe weight matrix + bias and returns it
    n = len(weightMatrix)
    if debug == 1:
        print("Input --> Hidden")
        print("Before: ",inputLayer)
        
    for i in range(n):
        #Forwarward Propogate from input to hidden
        #This grabs the value of the individual hidden layer node at index I at toNode
        A, Z = forwProp.activationInit(inputLayer, i, weightMatrix, bias)
        nextLayer.append(A)
        ZArray.append(Z)
    if debug == 1:   
        print("After: ",nextLayer)
   
    
def forwardPropogateLayer(inputLayer, weightMatrix, bias, nextLayer, ZArray, debug):
    
    #This function takes the sum of inpt nodesXthe weight matrix + bias and returns it
    n = len(weightMatrix)
    if debug == 1:
        print("Hidden --> Output")
        print("Before: ",inputLayer)
    
    for i in range(7):
        #Forwarward Propogate from input to hidden
        #This grabs the value of the individual hidden layer node at index I at toNode
        A, Z = forwProp.activationLayers(inputLayer, i, weightMatrix, bias)
        nextLayer.append(A)    
        ZArray.append(Z)
    if debug == 1:    
        print("After: ",nextLayer)
        print("\n")
            
            

class forwProp:

    def sigmoid(Z):
        return np.tanh(Z)
    
    def weightInit(value):
        return 1/np.sqrt(value)
    
     
    
    def cost(resultList, outputLayer, expectedCharacter):
        
        Actual = sum(outputLayer)
        expectedTotal = 1
        
        costError = pow((Actual - expectedTotal),2)
        
        
        
        return costError

            
    
    #Length of row = # of neurons in the input layer
    #Length of col = # of neurons in hidden layer
    #Sum of each input node -> 1 hiddenLayer node
    #  activation(entireInputLayer, tonNode = nextLayerNode index, entier weight matrix, bias)
    def activationInit(inputLayer, toNode, weightMatrix, bias):
        sumOfValues = 0
        for i in range(63):
            sumOfValues += inputLayer[i] * weightMatrix[toNode][i]
        
        Z = sumOfValues + bias[toNode]
        
        
        nextNodeValue = forwProp.sigmoid(Z)
        return nextNodeValue, Z
    
    def activationLayers(inputLayer, toNode, weightMatrix, bias):
        sumOfValues = 0
        
        n = len(inputLayer)
        #print("Tonode", toNode, "I", n)
        #print("len: ", len(weightMatrix),"x", len(weightMatrix[0]))
        
        for i in range(n):
            sumOfValues += inputLayer[i] * weightMatrix[toNode][i]
        
        Z = sumOfValues + bias[toNode]
        
        nextNodeValue = forwProp.sigmoid(Z)
        return nextNodeValue, Z
    
    #def propToLayer(inputLayer, toNode, weightMatrix, bias):
    

        
    
    
    
class Layer:
    
    def __init__(self, WeightMatrix, BiasMatrix, ZVals, AVals):
        self.WeightMatrix = WeightMatrix
        self.BiasMatrix = BiasMatrix
        self.ZVals = ZVals
        self.AVals = AVals
    
    def sigmoid(Z):
        return 1 / (1 + math.exp(-Z))
    
    def derivSigmoid(Z):
        return Layer.sigmoid(Z) * (1 - Layer.sigmoid(Z))
    
    def relu(Z):
        return max(0, Z)
    
    def relu_derivative(Z):
        return 1 if Z > 0 else 0


    
    def derivCostFunc(predicted, Actual):
        return 2*(predicted - Actual)
    
    def calculateWeight(A,Z, Cost):
        #print("A: ",A, "DerivSigmoid: ",Z ,"DerivCost: ", Cost)
        #print("Z", Z)
        return A*Layer.derivSigmoid(Z)*Cost
    
    def calculateBias(Cost, Z):
        return Layer.derivSigmoid(Z)*Cost
    
    def backWardPropogate(self, toLayer, cost):
        # print("len", len(self.AVals))
        # print("2#: ", len(toLayer.AVals))
        #print("\n\n\n\n")
        
        #format(self.WeightMatrix)
        
        #print("\n\n\n\n")
        for i in range(len(self.ZVals)):
            for j in range(len(toLayer.AVals)):
                
                newWeight = Layer.calculateWeight(toLayer.AVals[j], self.ZVals[i], cost) * .1
                
                newBias = Layer.calculateBias(cost, self.ZVals[i]) * .1
                
                self.WeightMatrix[i][j] = self.WeightMatrix[i][j] - newWeight
                
                self.BiasMatrix[i] = self.BiasMatrix[i] - newBias
                
        #format(self.WeightMatrix)
                
                
                 
                
    

        
    
    
main()
