import ROOT
import numpy as np
from sklearn.neural_network import MLPClassifier

# Number of samples for training the classifier
n_samples = 5000
n_samples_train = n_samples * 9  # Increase the training sample size for better accuracy

def make_likelihood(name, title, func, variables):
    """
    Create a custom likelihood function in ROOT using a Python-defined function.
    
    Parameters:
    - name (str): Name of the likelihood function.
    - title (str): Title of the likelihood function.
    - func (callable): Function that defines the likelihood.
    - variables (ROOT.RooArgList): List of variables to be used in the likelihood function.
    
    Returns:
    - MyLlh: A custom likelihood class derived from RooPyLikelihood.
    """
    class MyLlh(ROOT.RooPyLikelihood):
        def __init__(self, name, title, variables):
            super(MyLlh, self).__init__(name, title, ROOT.RooArgList(variables))

        def evaluate(self):
            return func(*(v.getVal() for v in self.varlist()))

        def clone(self, newname=False):
            cl = MyLlh(newname if newname else self.GetName(), self.GetTitle(), self.varlist())
            ROOT.SetOwnership(cl, False)
            return cl

    return MyLlh(name, title, variables)


class SBI:
    """
    Class for Simulation-Based Inference (SBI) using neural networks.
    
    Attributes:
    - classifier (MLPClassifier): A scikit-learn neural network classifier.
    - data_model (np.ndarray): Model data for training.
    - data_ref (np.ndarray): Reference data for training.
    - X_train (np.ndarray): Combined data (model + reference) for training.
    - y_train (np.ndarray): Labels for the training data.
    - ws (ROOT.RooWorkspace): ROOT workspace containing models and datasets.
    - n_vars (int): Number of variables (dimensions).
    - _training_mus (np.ndarray): Array of model parameters (mu) for training data.
    - _reference_mu (np.ndarray): Array of reference parameters (mu) for reference data.
    """
    def __init__(self, ws, n_vars):
        self.classifier = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        self.data_model = None
        self.data_ref = None
        self.X_train = None
        self.y_train = None
        self.ws = ws
        self.n_vars = n_vars
        self._training_mus = None
        self._reference_mu = None

    def model_data(self, model, x_vars, mu_vars, n_samples):
        """
        Generate model data using a Gaussian PDF and store it for training.
        
        Parameters:
        - model (str): Name of the Gaussian model in the workspace.
        - x_vars (list): List of observable variable names.
        - mu_vars (list): List of model parameter names.
        - n_samples (int): Number of samples to generate.
        """
        ws = self.ws
        samples_gaussian = ws[model].generate([ws[x] for x in x_vars] + [ws[mu] for mu in mu_vars], n_samples).to_numpy()

        self._training_mus = np.array([samples_gaussian[mu] for mu in mu_vars]).T
        data_test_model = np.array([samples_gaussian[x] for x in x_vars]).T

        self.data_model = data_test_model.reshape(-1, self.n_vars)

    def reference_data(self, model, x_vars, mu_vars, n_samples, help_model):
        """
        Generate reference data using a uniform distribution and store it for training.
        
        Parameters:
        - model (str): Name of the uniform model in the workspace.
        - x_vars (list): List of observable variable names.
        - mu_vars (list): List of model parameter names.
        - n_samples (int): Number of samples to generate.
        - help_model (str): Name of the helper uniform model for mu variables.
        """
        ws = self.ws

        samples_uniform = ws[model].generate([ws[x] for x in x_vars], n_samples)
        data_reference_model = np.array([samples_uniform.get(i).getRealValue(x) for x in x_vars for i in range(samples_uniform.numEntries())])

        self.data_ref = data_reference_model.reshape(-1, self.n_vars)

        samples_mu = ws[help_model].generate([ws[mu] for mu in mu_vars], n_samples)
        mu_data = np.array([samples_mu.get(i).getRealValue(mu) for mu in mu_vars for i in range(samples_mu.numEntries())])

        self._reference_mu = mu_data.reshape(-1, self.n_vars)

    def preprocessing(self):   
        """
        Preprocess the data by concatenating model and reference data.
        Also, prepare the labels for the classifier.
        """
        thetas = np.concatenate((self._training_mus, self._reference_mu))
        X = np.concatenate([self.data_model, self.data_ref])

        self.y_train = np.concatenate([np.ones(len(self.data_model)), np.zeros(len(self.data_ref))])
        self.X_train = np.concatenate([X, thetas], axis=1)

    def train_classifier(self):
        """
        Train the neural network classifier using the preprocessed data.
        """
        self.classifier.fit(self.X_train, self.y_train)


def build_ws(mu_observed):
    """
    Build a ROOT workspace with Gaussian and uniform distributions based on observed data.
    
    Parameters:
    - mu_observed (list): List of observed mean values (mu) for the Gaussian distributions.
    
    Returns:
    - ws (ROOT.RooWorkspace): Workspace containing the models and data.
    - x_vars (list): List of RooRealVar objects representing observable variables.
    - mu_vars (list): List of RooRealVar objects representing model parameters.
    """
    n_vars = len(mu_observed)

    # Define the observable variables and model parameters
    x_vars = [ROOT.RooRealVar(f"x{i}", f"x{i}", -12, 12) for i in range(n_vars)]
    mu_vars = [ROOT.RooRealVar(f"mu{i}", f"mu{i}", mu_observed[i], -4, 4) for i in range(n_vars)]
    sigma_vars = [ROOT.RooRealVar(f"sigma{i}", f"sigma{i}", 1.5, 0.1, 10) for i in range(n_vars)]

    for sigma in sigma_vars:
        sigma.setConstant()

    # Create Gaussian and uniform PDFs for each variable
    gaussians = [ROOT.RooGaussian(f"gauss{i}", f"gauss{i}", x_vars[i], mu_vars[i], sigma_vars[i]) for i in range(n_vars)]
    uniforms = [ROOT.RooUniform(f"uniform{i}", f"uniform{i}", x_vars[i]) for i in range(n_vars)]
    uniforms_help = [ROOT.RooUniform(f"uniformh{i}", f"uniformh{i}", mu_vars[i]) for i in range(n_vars)]

    # Create multi-dimensional PDFs
    gauss = ROOT.RooProdPdf("gauss", "gauss", ROOT.RooArgList(*gaussians))
    uniform = ROOT.RooProdPdf("uniform", "uniform", ROOT.RooArgList(*uniforms))
    uniform_help = ROOT.RooProdPdf("uniform_help", "uniform_help", ROOT.RooArgList(*uniforms_help))
    obs_data = gauss.generate(ROOT.RooArgSet(*x_vars), n_samples)
    obs_data.SetName("obs_data")

    # Create and return the workspace
    ws = ROOT.RooWorkspace()
    ws.Import(gauss)
    ws.Import(uniform)
    ws.Import(gaussians)
    ws.Import(uniform_help)
    ws.Import(obs_data)

    return ws, x_vars, mu_vars

# Define the observed mean values for the Gaussian distributions
mu_observed = [2.0, 1.0, 1.0, 1.0]

# Build the workspace and extract variables
ws, x_vars, mu_vars = build_ws(mu_observed)
ws.Print()

# Initialize the SBI model
model = SBI(ws, len(mu_observed))

# Generate and preprocess training data
model.model_data("gauss", [x.GetName() for x in x_vars], [mu.GetName() for mu in mu_vars], n_samples_train)
model.reference_data("uniform", [x.GetName() for x in x_vars], [mu.GetName() for mu in mu_vars], n_samples_train, "uniform_help")
model.preprocessing()

# Train the neural network classifier
model.train_classifier()
sbi_model = model

# Function to compute the likelihood ratio using the trained classifier
def compute_likelihood_ratio(*args):
    """
    Compute the likelihood ratio for given input values using the trained classifier.
    
    Parameters:
    - args: Combined list of x values (observables) and mu values (model parameters).
    
    Returns:
    - float: The probability of the data point belonging to the target distribution.
    """
    x_vals = args[:len(mu_observed)]
    mu_vals = args[len(mu_observed):]
    
    # Concatenate x and mu_vals into a single data point
    data_point = np.array(list(x_vals) + list(mu_vals)).reshape(1, -1)
    
    # Get probability of the data point being from the target distribution
    prob = sbi_model.classifier.predict_proba(data_point)[:, 1]
    
    return prob[0]

# Function to compute the summed logarithmic likelihood ratio
def compute_log_likelihood_sum(*args):
    """
    Compute the summed negative logarithmic likelihood ratio for a given set of parameters.
    
    Parameters:
    - args: List of mu values (model parameters).
    
    Returns:
    - float: Summed negative logarithmic likelihood ratio.
    """
    mu_vals = args[:len(mu_observed)]
    obs_data_np = ws["obs_data"].to_numpy()
    x_data = np.array([obs_data_np[x.GetName()] for x in x_vars]).T
    mu_arr = np.tile(mu_vals, (ws["obs_data"].numEntries(), 1))
    data_points = np.hstack([x_data, mu_arr])
    prob = sbi_model.classifier.predict_proba(data_points)[:, 1]

    # Compute the summed negative logarithmic likelihood ratio
    return np.sum(np.log((1 - prob) / prob))

# Create combined variable list for ROOT
combined_vars = ROOT.RooArgList()
for var in x_vars + mu_vars:
    combined_vars.add(var)

# Create a custom likelihood ratio function using the trained classifier
llhr_learned = make_likelihood("MyLlh", "My Llh", compute_likelihood_ratio, combined_vars)

# Additional ROOT-related operations for plotting
llhr_calc_t = [ROOT.RooFit.RooConst(-1)]
for i in range(len(mu_observed)):
    llhr_calc_t.append(ROOT.RooFormulaVar(f"llhr_calc", "x[1] / (x[0] + x[1])", ROOT.RooArgList(ws[f"gauss{i}"], ws[f"uniform{i}"])))

llhr_calc = ROOT.RooProduct("prod", " * ".join(f"llhr_calc_{i}" for i in range(len(mu_observed))), ROOT.RooArgList(*llhr_calc_t))

llhr_calc_false = ROOT.RooFormulaVar("llhr_calc_false", "x[0] / (x[0] + x[1])", ROOT.RooArgList(ws["gauss"], ws["uniform"]))
nll_gauss = ws["gauss"].createNLL(ws["obs_data"])
nll_uniform = ws["uniform"].createNLL(ws["obs_data"])
nllr__calc = ROOT.RooFormulaVar("nllr__calc", "nllr__calc", "x[1]-x[0]", ROOT.RooArgList(nll_gauss, nll_uniform))
nllr_learned = make_likelihood("MyLlh", "My Llh", compute_log_likelihood_sum, ROOT.RooArgList(mu_vars))

# Plot the learned and analytical summed logarithmic likelihood
c1 = ROOT.TCanvas()
frame = mu_vars[0].frame(Title="Learned vs analytical summed logarithmic Likelihood", Range=(mu_vars[0].getMin(), mu_vars[0].getMax()))
nll_uniform.plotOn(frame, Name="uni")
nllr__calc.plotOn(frame, LineColor='y', Name="ratio")
nll_gauss.plotOn(frame, ShiftToZero=True, LineColor='g', LineStyle='--', Name="gauss")
nllr_learned.plotOn(frame, LineColor="r", ShiftToZero=True, Name="learned")
frame.Draw()

# Add a legend to the plot
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry("uni", "Uniform", "l")
legend.AddEntry("ratio", "nll_u-nll_g", "l")
legend.AddEntry("gauss", "Gaussian", "l")
legend.AddEntry("learned", "sum(log((1 - prob)/prob))", "l")
legend.Draw()
c1.SaveAs("Logarithmic_summed_multidim.png")

# Declare a helper function in ROOT to dereference unique_ptr
ROOT.gInterpreter.Declare("""
RooAbsArg &my_deref(std::unique_ptr<RooAbsArg> const& ptr) { return *ptr; }
""")

# Compile for normalization and plot the likelihood functions
norm_set = ROOT.RooArgSet(x_vars)
llhr_calc_final_ptr = ROOT.RooFit.Detail.compileForNormSet(llhr_calc_false, norm_set)
llhr_calc_final = ROOT.my_deref(llhr_calc_final_ptr)
llhr_calc_final.recursiveRedirectServers(norm_set)

c2 = ROOT.TCanvas()
frame_x = x_vars[0].frame(Title="Learned vs analytical likelihood function")
llhr_learned.plotOn(frame_x, LineColor="r", Name="learned")
llhr_calc_final.plotOn(frame_x, LineColor="c", Name="analytical")
frame_x.Draw()

# Add a legend to the plot
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry("learned", "learned", "l")
legend.AddEntry("analytical", "analytical", "l")
legend.Draw()
c2.SaveAs("llh_function_multidim.png")

# Use ROOT's minimizer to compute the minimum and display the results
for nll in [nllr_learned, nll_gauss]:
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setErrorLevel(0.5)
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()
