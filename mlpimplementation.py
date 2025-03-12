import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                y_pred = self.predict_single(x_i)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) + self.bias > 0 else 0
                
    def predict(self, X):
        return np.array([self.predict_single(x_i) for x_i in X])





class MultiLayerPerceptronXOR:
    def __init__(self, learning_rate=0.1, epochs=100):
        #Create two "hidden" perceptrons 
        self.hidden_perceptron1 = Perceptron(learning_rate, epochs)
        self.hidden_perceptron2 = Perceptron(learning_rate, epochs)
        #Create one perceptron for output layer
        self.output_perceptron = Perceptron(learning_rate, epochs)
        
    def fit(self, X, y):
        #[0,1,0,0]
        #(0,1)
        h1_y = np.array([0, 1, 0, 0])
        self.hidden_perceptron1.fit(X, h1_y)
        
        #[0,0,1,0]
        #(1,0)
        h2_y = np.array([0, 0, 1, 0])
        self.hidden_perceptron2.fit(X, h2_y)
        
        #Get outputs
        h1_outputs = self.hidden_perceptron1.predict(X)
        h2_outputs = self.hidden_perceptron2.predict(X)
        
        #Combine
        X_final = np.column_stack((h1_outputs, h2_outputs))
        
        #training the output
        self.output_perceptron.fit(X_final, y)
    
    def predict(self, X):
        #Get outputs
        h1_outputs = self.hidden_perceptron1.predict(X)
        h2_outputs = self.hidden_perceptron2.predict(X)
        X_final = np.column_stack((h1_outputs, h2_outputs))
        
        return self.output_perceptron.predict(X_final)

        #XORExample
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])#XORtruth





#Create & train
mlp = MultiLayerPerceptronXOR(learning_rate=0.1, epochs=100)
mlp.fit(X, y)

predictions = mlp.predict(X)
print("Inputs:")
print(X)
print("\nPredictions:")
print(predictions)
print("\nExpected outputs:")
print(y)
print("\nHidden Perceptron 1 (detects the pattern [0,1]):")
print(f"Weights: {mlp.hidden_perceptron1.weights}, Bias: {mlp.hidden_perceptron1.bias}")
print("\nHidden Perceptron 2 (detects the pattern [1,0]):")
print(f"Weights: {mlp.hidden_perceptron2.weights}, Bias: {mlp.hidden_perceptron2.bias}")
print("\nOutput Perceptron (combines hidden layer outputs):")
print(f"Weights: {mlp.output_perceptron.weights}, Bias: {mlp.output_perceptron.bias}")
