import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from scipy.stats import norm, uniform
import ROOT

class SBI:

    # initializing the class SBI
    def __init__(self, workspace, mu_vals):
        # Choose the hyperparameters for training the neural network
        self.classifier = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        self.mu_vals = mu_vals
        self.data_model = None
        self.data_ref = None
        self.X_train = None
        self.y_train = None
        self.workspace = workspace

    # defining the target / training data for different values of mean value mu 
    def model_data(self, model, x, mu, n_samples):
        ws = self.workspace
        data_test_model = []
        # Preventing the old mu value from overwriting
        old_val = ws[mu].getVal()

        # getting training data for each value of mu
        for theta in self.mu_vals:
            ws[mu].setVal(theta)
            samples_gaussian = ws[model].generate(ws[x], n_samples)
            data_test_model.extend([samples_gaussian.get(i).getRealValue("x") for i in range(samples_gaussian.numEntries())])
        ws[mu].setVal(old_val)
        self.data_model = np.array(data_test_model).reshape(-1, 1)

    # generating samples for the reference distribution 
    def reference_data(self, model, x, n_samples):
        ws = self.workspace
        # ensuring the normalization with generating as many reference data as target data
        samples_uniform = ws[model].generate(ws[x], n_samples * len(self.mu_vals))
        data_reference_model = np.array([samples_uniform.get(i).getRealValue("x") for i in range(samples_uniform.numEntries())])
        self.data_ref = data_reference_model.reshape(-1, 1)

    # bringing the data in the right format for training
    def preprocessing(self):
        repeats_model = len(self.data_model) // len(self.mu_vals)
        thetas_model = np.repeat(self.mu_vals, repeats_model).reshape(-1, 1)
        repeats_reference = len(self.data_ref) // len(self.mu_vals)
        thetas_reference = np.repeat(self.mu_vals, repeats_reference).reshape(-1, 1)
        thetas = np.concatenate((thetas_model, thetas_reference), axis=0)
        X = np.concatenate([self.data_model, self.data_ref])
        self.y_train = np.concatenate([np.zeros(len(self.data_model)), np.ones(len(self.data_ref))])
        self.X_train = np.concatenate([X, thetas], axis=1)

    # train the classifier
    def train_classifier(self):
        self.classifier.fit(self.X_train, self.y_train)

    # calculate negative log ratio
    def calc_neg_log_ratio(self, x_vals):
        preds = []
        for x in x_vals:
            row_preds = []
            for mu in self.mu_vals:
                sample = np.array([[x, mu]])
                pred = self.classifier.predict_proba(sample)[:, 0]
                row_preds.append(-np.log(pred))
            preds.append(row_preds)
        return np.array(preds)
"""
    # calculate negative log ratio
    def calc_neg_log_ratio(self, x, mu_observed):
        preds = []
        for mu in self.mu_vals:
            sample = np.array([[x, mu]])
            pred = self.classifier.predict_proba(sample)[:, 0]
            preds.append(-np.log(pred))
        return preds
"""


# Setting the training and toy data samples 
n_samples_train = 1000

# The "observed" data 
mu_observed = 1.

# define the "observed" data
x_var = ROOT.RooRealVar("x", "x", -12, 12)
mu_var = ROOT.RooRealVar("mu", "mu", mu_observed, -12, 12)
sigma_var = ROOT.RooRealVar("sigma", "sigma", 1.5, 0.1, 10)
gauss = ROOT.RooGaussian("gauss", "gauss", x_var, mu_var, sigma_var)
uniform_r = ROOT.RooUniform("uniform", "uniform", x_var)
obs_data = gauss.generate(x_var, n_samples_train)
obs_data_u = uniform_r.generate(x_var, n_samples_train)

# using a workspace for easier processing inside the class
workspace = ROOT.RooWorkspace()
workspace.Import(gauss)
workspace.Import(uniform_r)
workspace.Import(obs_data)
workspace.Print()

# range of mean values for plotting
mu_vals = [-3,1,3]

# training the model 
model = SBI(workspace, mu_vals)
model.model_data("gauss", "x", "mu", n_samples_train)
model.reference_data("uniform", "x", n_samples_train)
model.preprocessing()
model.train_classifier()
sbi_model = model

# Calculate negative log ratio using the theoretical method
sigma = 1.5  # standard deviation for the Gaussian distribution
a, b = -12, 12  # bounds for the Uniform distribution
x = 0  # the value at which to evaluate the PDFs
x_linspace = np.linspace(-5, 5, 100)

neg_log_ratios_theoretical = []
for mu in x_linspace:
    gaussian_pdf = norm.pdf(x, loc=mu, scale=sigma)
    uniform_pdf = uniform.pdf(x, loc=a, scale=(b-a))
    ratio = gaussian_pdf / uniform_pdf
    neg_log_ratio = -np.log(ratio)
    neg_log_ratios_theoretical.append(neg_log_ratio)

# Calculate negative log ratio using the SBI model
neg_log_ratios_sbi = sbi_model.calc_neg_log_ratio(x_linspace)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_linspace, neg_log_ratios_theoretical, label='Theoretical Negative Log Ratio')
plt.plot(x_linspace, neg_log_ratios_sbi, label='SBI Negative Log Ratio', linestyle='--')
plt.xlabel('Mean (μ)')
plt.ylabel('Negative Log Ratio')
plt.title('Negative Log Ratio of Gaussian vs Uniform Distribution')
plt.legend()
plt.grid(True)
plt.show()
