            self.gradients[f'dW{i+1}'] = np.clip(self.gradients[f'dW{i+1}'])
            self.gradients[f'db{i+1}'] = np.clip(self.gradients[f'db{i+1}'])