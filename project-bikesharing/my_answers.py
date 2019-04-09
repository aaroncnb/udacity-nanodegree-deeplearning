import numpy as np


class NeuralNetwork(object):
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set _function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        
        #self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(Z):
            return 1/(1+np.exp(-Z))  # Replace 0 with your sigmoid calculation here

        def sigmoid_grad(dA):
            return dA*(1-dA)  # Replace 0 with your sigmoid calculation here
        # Define some alternate activaitons-- 
        # these are no-brainers, but writing them anyways to keep things explicit

        def linear(Z):  # Simplest activation when unbounded outputs are needed
            return Z
        
        def linear_grad(dZ):
            return np.ones_like(dZ)

        def exponential(Z): # One option for requiring positive-only outputs
            return np.exp(Z)
        
        def exponential_grad(dZ):
            return dZ

        def relu(x):
            return x*(x>0)

        def relu_grad(x):
            return 1. * (x>0)

        self.activation_function       = sigmoid # Choose the activation
        self.activation_grad           = sigmoid_grad
        
        # self.activation_function_final = exponential  # Choose the output activation
        # self.activation__grad_final    = exponential_grad        
        self.activation_function_final = linear # Choose the output activation
        self.activation__grad_final    = linear_grad 
        # self.activation_function_final = relu # Choose the output activation
        # self.activation__grad_final    = relu_grad       
        
    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
       
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        
        #print("X shape = {}".format(X.shape))
        #print("W_IH shape = {}".format(self.weights_input_to_hidden.shape))
            
        hidden_inputs = np.dot(X.T, self.weights_input_to_hidden) # signals into hidden layer
        #print("Z_h shape = {}".format(hidden_inputs.shape))
            
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        #if debug == True: print("A_h shape = {}".format(hidden_outputs.shape))

        # TODO: Output layer - Replace these values with your calculations.
        #if debug == True: print("W_HO shape = {}".format(self.weights_hidden_to_output.shape))
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        
        #if debug == True: print("Z_o shape = {}".format(final_inputs.shape))
        final_outputs = self.activation_function_final(final_inputs)
        #final_outputs = self.activation_function(final_inputs) # signals from final output layer
        #if debug == True: print("A_o shape = {}".format(final_outputs.shape))
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        #print("y shape = {}".format(y.shape))
        #print("A_o sha")
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        #if self.debug== True: print("error shape = {}".format(error.shape))
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        #A_o_grad = self.activation_grad(final_outputs)
        A_o_grad = self.activation__grad_final(final_outputs) # 1 if the output layer was linear
        #if self.debug== True: print("A_o_grad shape = {}".format(A_o_grad.shape))
        
        output_error_term = error * A_o_grad.T
        #if self.debug== True: print("A_o_error shape = {}".format(output_error_term.shape))
        
        # TODO: Calculate the hidden layer's contribution to the error
        #print("")
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        
        #if self.debug== True: print("A_h_error = {}".format(hidden_error.shape))
        
        A_h_grad = self.activation_grad(hidden_outputs)
        #if self.debug== True: print("A_h_grad shape = {}".format(A_h_grad.shape))
        
        hidden_error_term = hidden_error * A_h_grad.T
        
        # Weight step (input to hidden)
        delta_weights_i_h +=  hidden_error_term * X[:,None]
        # Weight step (hidden to output)
        delta_weights_h_o +=  output_error_term * hidden_outputs[:,None]
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs.copy() # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2000
learning_rate = 1.0
hidden_nodes = 10 
output_nodes = 1
