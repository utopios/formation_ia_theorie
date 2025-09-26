import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class MiniLSTM:
    def __init__(self, W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o):
        self.W_f = np.array(W_f)
        self.b_f = b_f
        self.W_i = np.array(W_i)
        self.b_i = b_i
        self.W_C = np.array(W_C)
        self.b_C = b_C
        self.W_o = np.array(W_o)
        self.b_o = b_o
    
    def forward_step(self, h_prev, C_prev, x):
        input_concat = np.array([h_prev, x])
        
        z_f = np.dot(self.W_f, input_concat) + self.b_f
        f = sigmoid(z_f)
        
        z_i = np.dot(self.W_i, input_concat) + self.b_i
        i = sigmoid(z_i)
        
        z_C = np.dot(self.W_C, input_concat) + self.b_C
        C_tilde = tanh(z_C)
        
        C = f * C_prev + i * C_tilde
        
        z_o = np.dot(self.W_o, input_concat) + self.b_o
        o = sigmoid(z_o)
        
        h = o * tanh(C)
        
        return h, C, f, i, C_tilde, o
    
    def forward_sequence(self, sequence, h_0=0.0, C_0=0.0):
        h_prev, C_prev = h_0, C_0
        results = []
        
        for t, x in enumerate(sequence, 1):
            h, C, f, i, C_tilde, o = self.forward_step(h_prev, C_prev, x)
            results.append({
                'timestep': t,
                'input': x,
                'forget_gate': f,
                'input_gate': i,
                'candidate': C_tilde,
                'cell_state': C,
                'output_gate': o,
                'hidden_state': h
            })
            h_prev, C_prev = h, C
        
        return results

class LSTM2Neurons:
    def __init__(self, W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o):
        self.W_f = np.array(W_f)
        self.b_f = np.array(b_f)
        self.W_i = np.array(W_i)
        self.b_i = np.array(b_i)
        self.W_C = np.array(W_C)
        self.b_C = np.array(b_C)
        self.W_o = np.array(W_o)
        self.b_o = np.array(b_o)
    
    def forward_step(self, h_prev, C_prev, x):
        input_concat = np.concatenate([h_prev, [x]])
        
        z_f = np.dot(self.W_f, input_concat) + self.b_f
        f = sigmoid(z_f)
        
        z_i = np.dot(self.W_i, input_concat) + self.b_i
        i = sigmoid(z_i)
        
        z_C = np.dot(self.W_C, input_concat) + self.b_C
        C_tilde = tanh(z_C)
        
        C = f * C_prev + i * C_tilde
        
        z_o = np.dot(self.W_o, input_concat) + self.b_o
        o = sigmoid(z_o)
        
        h = o * tanh(C)
        
        return h, C, f, i, C_tilde, o
    
    def forward_sequence(self, sequence, h_0, C_0):
        h_prev, C_prev = np.array(h_0), np.array(C_0)
        results = []
        
        for t, x in enumerate(sequence, 1):
            h, C, f, i, C_tilde, o = self.forward_step(h_prev, C_prev, x)
            results.append({
                'timestep': t,
                'input': x,
                'forget_gate': f,
                'input_gate': i,
                'candidate': C_tilde,
                'cell_state': C,
                'output_gate': o,
                'hidden_state': h
            })
            h_prev, C_prev = h, C
        
        return results

def run_mini_lstm_example():
    lstm = MiniLSTM(
        W_f=[0.1, 0.2], b_f=0.0,
        W_i=[0.3, -0.1], b_i=0.1,
        W_C=[0.2, 0.4], b_C=0.0,
        W_o=[0.1, 0.3], b_o=0.0
    )
    
    sequence = [1.0, 0.5, -0.3]
    results = lstm.forward_sequence(sequence)
    
    print("Mini-LSTM Results:")
    for result in results:
        print(f"t={result['timestep']}: h={result['hidden_state']:.4f}, C={result['cell_state']:.4f}")
    
    return results

def run_lstm_2neurons_example():
    W_f = [[0.1, 0.2, 0.3], [0.0, 0.1, -0.2]]
    b_f = [0.1, 0.0]
    W_i = [[0.2, -0.1, 0.4], [0.1, 0.3, 0.1]]
    b_i = [0.0, 0.1]
    W_C = [[0.3, 0.1, -0.1], [-0.1, 0.2, 0.3]]
    b_C = [0.05, 0.0]
    W_o = [[0.1, 0.4, 0.2], [0.2, 0.0, 0.3]]
    b_o = [0.1, 0.0]
    
    lstm = LSTM2Neurons(W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o)
    
    sequence = [0.8]
    h_0 = [0.1, -0.1]
    C_0 = [0.2, 0.0]
    
    results = lstm.forward_sequence(sequence, h_0, C_0)
    
    print("\nLSTM 2-Neurons Results:")
    for result in results:
        h = result['hidden_state']
        C = result['cell_state']
        print(f"t={result['timestep']}: h=[{h[0]:.4f}, {h[1]:.4f}], C=[{C[0]:.4f}, {C[1]:.4f}]")
    
    return results

def main():
    mini_results = run_mini_lstm_example()
    lstm2_results = run_lstm_2neurons_example()
    
    return mini_results, lstm2_results

if __name__ == "__main__":
    main()