## \file
## \ingroup tutorial_roofit
## \notebook
## Use Morphing in RooFit
##
## This tutorial shows how to use morphing inside RooFit. As input we have several
## gaussian distributions. The output is one gaussian, with a specific mean value.
## Since likelihoods are often used within the framework of morphing, we provide a
## way to estimate the negative log likelihood (nll).
##
## \macro_code
## \macro_output
##
## \date August 2024
## \author Robin Syring


import ROOT
import numpy as np
from sklearn.neural_network import MLPClassifier

# The samples used to fill the histograms, adjust it to your needs
n_samples = 100


# morphing as a baseline
def morphing(setting):
    # set up a frame for plotting
    frame1 = x_var.frame()

    # define binning for morphing
    bin_mu_x = ROOT.RooBinning(4, 0.0, 4.0)
    grid = ROOT.RooMomentMorphFuncND.Grid2(bin_mu_x)
    x_var.setBins(50)

    # number of 'sampled' gaussians, if you change it, adjust the binning properly
    parampoints = np.arange(5)

    for i in parampoints:
        # define the sampled gausians
        mu_help = ROOT.RooRealVar(f"mu{i}", f"mu{i}", i)
        help = ROOT.RooGaussian(f"g{i}", f"g{i}", x_var, mu_help, sigma_var)
        workspace.Import(help)

        # fill the histograms
        hist = workspace[f"g{i}"].generateBinned([x_var], n_samples)

        # make sure that every bin is filled and we don't get zero probability
        for i_bin in range(hist.numEntries()):
            hist.add(hist.get(i_bin), 1.0)

        # add the pdf to the workspace, the inOrder of 1 is necessary for calculation oif the nll
        # adjust it to 0 to see binning
        workspace.Import(ROOT.RooHistPdf(f"histpdf{i}", f"histpdf{i}", [x_var], hist, intOrder=1))
        # pdf = workspace[f'histpdf{i}']

        # add the pdf to the grid
        grid.addPdf(workspace[f"histpdf{i}"], int(i))

        # for plotting purposes
        workspace[f"histpdf{i}"].plotOn(frame1)

    # create the morphing and add it to the workspace
    morph_func = ROOT.RooMomentMorphFuncND("morph_func", "morph_func", [mu_var], [x_var], grid, setting)

    # normalizes the morphed object to be a pdf, set it false to prevent warning messages and computational speed up
    morph_func.setPdfMode()

    # creating the morphed pdf
    morph = ROOT.RooWrapperPdf("morph", "morph", morph_func, True)
    workspace.Import(morph)
    workspace["morph"].plotOn(frame1, LineColor="r")

    c0 = ROOT.TCanvas()
    frame1.Draw()

    # the input prevents the plot from beeing closed instantly
    input("")


# define the "observed" data in a workspade
def build_ws(mu_observed):
    x_var = ROOT.RooRealVar("x", "x", -5, 15)
    mu_var = ROOT.RooRealVar("mu", "mu", mu_observed, 0, 4)
    sigma_var = ROOT.RooRealVar("sigma", "sigma", 1.5)
    gauss = ROOT.RooGaussian("gauss", "gauss", x_var, mu_var, sigma_var)
    uniform = ROOT.RooUniform("uniform", "uniform", x_var)
    obs_data = gauss.generate(x_var, n_samples)
    obs_data.SetName("obs_data")

    # using a workspace for easier processing inside the class
    workspace = ROOT.RooWorkspace()
    workspace.Import(x_var)
    workspace.Import(mu_var)
    workspace.Import(gauss)
    workspace.Import(uniform)
    workspace.Import(obs_data)

    return workspace


# The "observed" data
mu_observed = 2.5
workspace = build_ws(mu_observed)
x_var = workspace["x"]
mu_var = workspace["mu"]
sigma_var = workspace["sigma"]
gauss = workspace["gauss"]
uniform = workspace["uniform"]
obs_data = workspace["obs_data"]


# compute the real likelihood ration
llhr_calc = ROOT.RooFormulaVar("llhr_calc", "x[0] / (x[0] + x[1])", [gauss, uniform])

# Create the exact negative log likelihood functions for Gaussian model
nll_gauss = gauss.createNLL(obs_data)

# compute the morphed nll
morphing(ROOT.RooMomentMorphFuncND.Linear)
nll_morph = workspace["morph"].createNLL(obs_data)

# Plot the negative logarithmic summed likelihood
c1 = ROOT.TCanvas()
frame = mu_var.frame(Title="SBI vs. Morphing")
nll_gauss.plotOn(frame, LineColor="b", ShiftToZero=True, Name="gauss")
nll_morph.plotOn(frame, LineColor="r", ShiftToZero=True, Name="morphed")
frame.Draw()

# Create a legend and add entries
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Adjust coordinates as needed
legend.AddEntry("gauss", "Gaussian", "l")
legend.AddEntry("learned", "SBI", "l")
legend.AddEntry("morphed", "Morphed", "l")
legend.Draw()


# Compute the minimum via minuit and display the results
for i in [nll_gauss, nll_morph]:
    min = minimizer = ROOT.RooMinimizer(i)
    minimizer.setErrorLevel(0.5)  # adjust the error level in the minimization to work with likelihoods
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()
