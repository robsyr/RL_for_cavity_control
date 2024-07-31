import ROOT
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# The samples used for training the classifier in this tutorial / rescale for more accuracy
n_samples = 1000

ROOT.gInterpreter.Declare("""
class RooPyLikelihood : public RooAbsReal {
public:
   RooPyLikelihood(const char *name, const char *title, RooArgList &varlist)
      : RooAbsReal(name, title), m_varlist("!varlist", "All variables(list)", this)
   {
      m_varlist.add(varlist);
   }
   // copy constructor
   RooPyLikelihood(const RooPyLikelihood &right, const char *name = nullptr)
      : RooAbsReal(right, name), m_varlist("!varlist", this, right.m_varlist)
   {
   }
   // virtual destructor
   virtual ~RooPyLikelihood() {}
   // clone method
   RooPyLikelihood *clone(const char *name) const override { return new RooPyLikelihood(*this, name); }
   // the actual evaluation of function (will be redefined in Python!)
   Double_t evaluate() const override { return 1; }
   // getter for varlist
   const RooArgList &varlist() const { return m_varlist; }
protected:
   RooListProxy m_varlist; // all variables as list of variables
};
""")

# Overwriting the cpp function RooPyLikelihood
def make_likelihood(name, title, func, variables):
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
    def __init__(self, workspace, n_vars):
        self.classifier = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        self.data_model = None
        self.data_ref = None
        self.X_train = None
        self.y_train = None
        self.workspace = workspace
        self.n_vars = n_vars

    def model_data(self, model, x_vars, mu_vars, n_samples):
        ws = self.workspace
        data_test_model = []

        # Save old mu values
        old_vals = [ws[mu].getVal() for mu in mu_vars]

        samples_gaussian_data = ws[model].generate([ws[x] for x in x_vars] + [ws[mu] for mu in mu_vars], n_samples)

        samples_gaussian = samples_gaussian_data.to_numpy()
        
        self._training_mus = np.array([samples_gaussian[mu] for mu in mu_vars]).T
        data_test_model.extend(np.array([samples_gaussian[x] for x in x_vars]).T)

        for i, mu in enumerate(mu_vars):
            ws[mu].setVal(old_vals[i])

        self.data_model = np.array(data_test_model).reshape(-1, self.n_vars)
        print(self.data_model)

    def reference_data(self, model, x_vars, n_samples):
        ws = self.workspace
        samples_uniform = ws[model].generate([ws[x] for x in x_vars], n_samples)
        data_reference_model = np.array([samples_uniform.get(i).getRealValue(x) for x in x_vars for i in range(samples_uniform.numEntries())])
        self.data_ref = data_reference_model.reshape(-1, self.n_vars)

    def preprocessing(self):
        thetas_model = self._training_mus
        thetas_reference = np.repeat(np.mean(self._training_mus, axis=0).reshape(1, -1), len(self.data_ref), axis=0)

        thetas = np.concatenate((thetas_model, thetas_reference), axis=0)
        X = np.concatenate([self.data_model, self.data_ref])

        self.y_train = np.concatenate([np.ones(len(self.data_model)), np.zeros(len(self.data_ref))])
        self.X_train = np.concatenate([X, thetas], axis=1)

    def train_classifier(self):
        self.classifier.fit(self.X_train, self.y_train)


# Number of variables (dimensions)
n_vars = 2

# Setting the training and toy data samples 
n_samples_train = n_samples * 9

# The "observed" data
mu_observed = [1.5, 1.0]

# Define the "observed" data
x_vars = [ROOT.RooRealVar(f"x{i}", f"x{i}", -12, 12) for i in range(n_vars)]
mu_vars = [ROOT.RooRealVar(f"mu{i}", f"mu{i}", mu_observed[i], -4, 4) for i in range(n_vars)]
sigma_vars = [ROOT.RooRealVar(f"sigma{i}", f"sigma{i}", 1.5, 0.1, 10) for i in range(n_vars)]
gaussians = [ROOT.RooGaussian(f"gauss{i}", f"gauss{i}", x_vars[i], mu_vars[i], sigma_vars[i]) for i in range(n_vars)]
uniforms = [ROOT.RooUniform(f"uniform{i}", f"uniform{i}", x_vars[i]) for i in range(n_vars)]



# Create RooProdPdf for multi-dimensional Gaussian and Uniform distributions
gauss = ROOT.RooProdPdf("gauss", "gauss", ROOT.RooArgList(*gaussians))
uniform = ROOT.RooProdPdf("uniform", "uniform", ROOT.RooArgList(*uniforms))
obs_data = gauss.generate(ROOT.RooArgSet(*x_vars), n_samples)


# Using a workspace for easier processing inside the class
workspace = ROOT.RooWorkspace()
workspace.Import(gauss)
workspace.Import(uniform)
workspace.Import(obs_data)
workspace.Print()

# Training the model 
model = SBI(workspace, n_vars)
model.model_data("gauss", [x.GetName() for x in x_vars], [mu.GetName() for mu in mu_vars], n_samples_train)
model.reference_data("uniform", [x.GetName() for x in x_vars], n_samples_train)
model.preprocessing()
model.train_classifier()
sbi_model = model


# Compute the likelihood ratio of the classifier for analysis purposes
def compute_likelihood_ratio(*args):
    x_vals = args[:n_vars]
    mu_vals = args[n_vars:]
    
    # Concatenate x and mu_vals into a single data point
    data_point = np.array(list(x_vals) + list(mu_vals)).reshape(1, -1)
    print("data_point:", data_point[0], data_point)
    
    # Get probability of the data point being from the target distribution
    prob = sbi_model.classifier.predict_proba(data_point)[:, 1]
    print(prob[0])
    
    return prob[0]


# Compute the negative logarithmic likelihood ratio summed
def compute_log_likelihood_sum(mu_vals):
    mu_arr = np.tile(mu_vals, (obs_data.numEntries(), 1))
    data_points = np.hstack([obs_data.to_numpy()[:, [x.GetName() for x in x_vars]], mu_arr])
    prob = sbi_model.classifier.predict_proba(data_points)[:, 1]
    return np.sum(np.log((1 - prob) / prob))

# Create combined variable list
combined_vars = ROOT.RooArgList()
for x in x_vars:
    combined_vars.add(x)
    print ('xs',x)
for mu in mu_vars:
    combined_vars.add(mu)

# Compute the likelihood ratio
llhr_learned = make_likelihood("MyLlh", "My Llh", compute_likelihood_ratio, combined_vars)

# Compute the real likelihood ratio
llhr_calc = ROOT.RooFormulaVar("llhr_calc", "x[0] / (x[0] + x[1])", ROOT.RooArgList(gauss, uniform))

# Create negative log likelihood functions for Gaussian and Uniform models
nll_gauss = gauss.createNLL(obs_data)
nll_uniform = uniform.createNLL(obs_data)

# Compute the "real" negative log likelihood
nllr__calc = ROOT.RooFormulaVar("nllr__calc", "nllr__calc", "x[1]-x[0]", ROOT.RooArgList(nll_gauss, nll_uniform))

# Compute the "learned" negative log likelihood ratio
nllr_learned = make_likelihood("MyLlh", "My Llh", lambda *x: compute_log_likelihood_sum(x[-n_vars:]), ROOT.RooArgList(*mu_vars))


"""
# Plot the negative logarithmic summed likelihood
c1 = ROOT.TCanvas()
frame = mu_vars[0].frame(Title="Learned vs analytical summed logarithmic Likelihood", Range=(mu_observed[0]-5, mu_vars[0].getMax()))
nll_uniform.plotOn(frame, Name="uni")
nllr__calc.plotOn(frame, LineColor='y', Name="ratio")
nll_gauss.plotOn(frame, ShiftToZero=True, LineColor='g', LineStyle='--', Name="gauss")
nllr_learned.plotOn(frame, LineColor="r", ShiftToZero=True, Name="learned")
frame.Draw()

# Create a legend and add entries
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry("uni", "Uniform", "l")
legend.AddEntry("ratio", "nll_u-nll_g", "l")
legend.AddEntry("gauss", "Gaussian", "l")
legend.AddEntry("learned", "sum(log((1 - prob)/prob))", "l")
legend.Draw()
c1.SaveAs("Logarithmic_summed_multidim.png")
"""
# Plot the likelihood functions
c2 = ROOT.TCanvas()
frame_x = x_vars[0].frame(Title="Learned vs analytical likelihood function")
llhr_learned.plotOn(frame_x, LineColor="r", Name="learned")
llhr_calc.plotOn(frame_x, Name="exact")
frame_x.Draw()

# Create a legend and add entries
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry("learned", "learned", "l")
legend.AddEntry("exact", "exact", "l")
legend.Draw()
c2.SaveAs("llh_function_multidim.png")

# Compute the minimum via minuit and display the results
for i in [nll_gauss, llhr_learned]: #, nllr_learned]:

    min = minimizer = ROOT.RooMinimizer(i)
    minimizer.setErrorLevel(0.5)
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()
