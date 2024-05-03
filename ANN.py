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
        W1 = [[rand.random() for i in range(63)] for j in range(numHidden)]
        B1 = [rand.random() for i in range(numHidden)]
        
        W2 = [[rand.random() for i in range(numHidden)] for j in range(7)]
        B2 = [rand.random() for i in range(7)]
        
        return lettersArray, W1, W2, B1, B2
    

def main():
    #Number of nodes in the hidden
    #Returns the initial weight matrix and Bias matrix
    inputLayer, W1, W2, B1, B2 = createData(60)

    hiddenLayer = []
    
    
    outputLayer = []
    
    correctValues = ["A","B","C","D","E","J","K"]
    
    
    
    #Sum all values from node to each middle layer node
    #Input Layer ----> hiddenLayer
    #Iterate over each example in the input Layer, contains each example
    #for letter in inputLayer:
    
    #initilized to input 0 for now
    forwardPropogateInit(inputLayer[0], W1, B1, hiddenLayer)
    
    #print(hiddenLayer)
    
    forwardPropogateLayer(hiddenLayer, W2, B2, outputLayer)
    
    #print(outputLayer)
   
    
    
    
    
    

    pass 

def forwardPropogateInit(inputLayer, weightMatrix, bias, nextLayer):
    
    #This function takes the sum of input nodesXthe weight matrix + bias and returns it
    n = len(weightMatrix)
    
    for i in range(n):
        #Forwarward Propogate from input to hidden
        #This grabs the value of the individual hidden layer node at index I at toNode
        nextLayer.append(forwProp.activationInit(inputLayer, i, weightMatrix, bias))
    
def forwardPropogateLayer(inputLayer, weightMatrix, bias, nextLayer):
    
    #This function takes the sum of inpt nodesXthe weight matrix + bias and returns it
    n = len(weightMatrix)
    
    for i in range(7):
        #Forwarward Propogate from input to hidden
        #This grabs the value of the individual hidden layer node at index I at toNode
        nextLayer.append(forwProp.activationLayers(inputLayer, i, weightMatrix, bias))       
    
            
            

class forwProp:

    def sigmoid(Z):
        return 1 / (1 + math.exp(-Z))

    def Z():
        pass
    def MSW():
        pass
    
    #####IndexOfInputLayer##############
    #I##################################
    #N##################################
    #D##################################
    #E##################################
    #X##################################
    #N##################################
    #E##################################
    #U##################################
    #R##################################
    #O##################################
    #Sum of each input node -> 1 hiddenLayer node
    #  activation(entireInputLayer, tonNode = nextLayerNode index, entier weight matrix, bias)
    def activationInit(inputLayer, toNode, weightMatrix, bias):
        sumOfValues = 0
        
        for i in range(63):
            sumOfValues += inputLayer[i] * weightMatrix[toNode][i]
        
        Z = sumOfValues + bias[toNode]
        
        nextNodeValue = forwProp.sigmoid(Z)
        return nextNodeValue
    
    def activationLayers(inputLayer, toNode, weightMatrix, bias):
        sumOfValues = 0
        n = len(inputLayer)
        for i in range(n):
            sumOfValues += inputLayer[i] * weightMatrix[toNode][i]
        
        Z = sumOfValues + bias[toNode]
        
        nextNodeValue = forwProp.sigmoid(Z)
        return nextNodeValue
    
    #def propToLayer(inputLayer, toNode, weightMatrix, bias):
    
    
    
        
def outputLayer(outPutLayer):
    
    
    pass
        
    
    
    
# class backProp:
#     def derivSigmoid():
#         pass
#     def derivZ():
#         pass
#     def derivMSE():
#         pass
      

    
    



    

    
    

    
    
    
    
    
    
    
    
main()
