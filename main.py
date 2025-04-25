import numpy as np
from MLP import MLP


def main():
    mlp = MLP(False)
    mlp.train()

main()

'''
def calc_dC_dW(self, L, j, k):
    
    Calculate the derivative of the Cost function with respect to the derivative
    of the weight WLjk, where
    L is the layer (not including inputs, so 1st hidden layer is 0)
    j is the index of the right side node
    k is the index of the left side node.
    
    The derivative is chain ruling 3 partial derivatives:
    1) dz/dw, where z is the linear func weight * activation + bias
    2) da/dz, where a is the activaion of the node after sigmoid(z)
    3) dC/da, where C is the cost
    
    Because of the nature of the dC/da term using previously calculated values,
    the calling function for this method should initialize a self.dC_da matrix to store
    values. (L, j) will reference the dC/da for node in layer L, jth node.
    EX: self.dC_da[0][2] will refence layer 0, the weights leading into the outputs,
    and the 3rd node down
    
    
    # The derivative of dz/dw is the activation of Node L,k
    dz_dw = 0.0
    if L > 0:
        dz_dw = self.hidden_nodes[L][k]
    else:
        dz_dw = self.input_nodes[k]
    
    # The derivative of da/dz is the the derivative of sigmoid(linear), which is sigmoid(linear) * (1 - sigmoid(linear))
    # which can also be said as activation * (1 - activation)
    da_dz = 0.0
    if L < Constants.HIDDEN_LAYER_COUNT:
        da_dz = self.hidden_nodes[L][j] * (1 - self.hidden_nodes[L][j])
    else:
        da_dz = self.output_nodes[j] * (1 - self.output_nodes[j])
        
    # The derivative of dC/da is more complicated... so I made a separate function for it
    dc_da: float = self.calc_dC_da(L, j)
    
    # Calculate the change we will make to the weight
    change = Constants.LEARNING_RATE * dz_dw * da_dz * dc_da
    if L == Constants.HIDDEN_LAYER_COUNT:
        self.delta_output_weights = change
    else:
        self.delta_hidden_weights = change
'''
        
        
            
            
'''
def calc_dC_da(self, L, k):
    ret = 0.0
    if L == Constants.HIDDEN_LAYER_COUNT:
        return self.dC_da_outputs[j]
    else:
        # Get an array of dc/da values for the nodes to the right
        target_dC_da_array = self.dC_da_outputs if L == (Constants.HIDDEN_LAYER_COUNT - 1) else self.dC_da[L + 1]
        
        # A nodes activation affects the Cost through all the nodes in the right layer, sinces its activation affect their activations
        # specifically we need to sum up for our node k, leading to each node j, add (dzL+1j/daLk * daL+1j/dzL+1j * dC/daL+1j)
        for j in range (target_dC_da_array.size):
            # First get dzL+1j/daLk, the derivative simplies to the weight between the 2 nodes
            dz_da = 0.0
            if (L == Constants.HIDDEN_LAYER_COUNT - 1):
                dz_da = self.output_weights[k][j]
            else:
                dz_da = self.hidden_weights[L+1][k][j]
            
            # Next get daL+1j/dzL+1j, which is derivative of sigmoid(z), which is a * (1 - a)
            da_dz = 0.0
            if (L == Constants.HIDDEN_LAYER_COUNT - 1):
                da_dz = self.output_nodes[j] * (1 - self.output_nodes[j])
            else:
                da_dz = self.hidden_nodes[L+1][j] * (1 - self.hidden_nodes[L+1][j])
            
            # Last we get the cost, which we are storing each step of the way
            dC_da = target_dC_da_array[j]
            
            # Now multiply them and add to ret
            ret += (dz_da * da_dz * dC_da)
        
        # populate the self.dC_da matrix with this calculated value then return it
        self.dC_da[L][k] = ret
        return ret
''' 