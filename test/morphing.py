import ROOT
import numpy as np

# Initialize workspace with some common background part
w = ROOT.RooWorkspace('w')
w.factory('Exponential::e(x[-5,15],tau[-.15,-3,0])')
x = w.var('x')
x.setBins(50)

frame = x.frame()

# Center of Gaussian will move along the parameter points
mu = w.factory('mu[0,4]')  # This is our continuous interpolation parameter
paramPoints = np.arange(5)

# Grid binning
bin_mu_x = ROOT.RooBinning(4, 0., 4.)
grid = ROOT.RooMomentMorphFuncND.Grid2(bin_mu_x)

# Now make the specific HistPdf from samples to add on top of common background
for i in paramPoints:
    # Create Gaussian distribution for sampling
    w.factory('Gaussian::g{i}(x,mu{i}[{i}],sigma[1])'.format(i=i))

    # Sample from the Gaussian to create histogram
    hist = w[f'g{i}'].generateBinned([x], 1000)  # Adjust the number of samples as needed
    print(i)


    # To make sure that every bin is filled and we don't get zero probability
    # in the template pdf (only needed for non-linear morphing)
    for i_bin in range(hist.numEntries()):
        hist.add(hist.get(i_bin), 1.0)


    w.Import(ROOT.RooHistPdf(f'histpdf{i}', f'histpdf{i}', ROOT.RooArgSet(x), hist))
    pdf = w[f'histpdf{i}']


    # Add the pdf to the grid
    grid.addPdf(pdf, int(i))


# Interpolation setting
setting = ROOT.RooMomentMorphFuncND.Linear

# Create the morphing function
morph_func = ROOT.RooMomentMorphFuncND('morph_func', 'morph_func', ROOT.RooArgList(mu), ROOT.RooArgList(x), grid, setting)
#morph_func._isPdfMode=True # workaround for normalizationset problem ?
# morph_func.setPdfMode()
morph = ROOT.RooWrapperPdf("morph", "morph", morph_func, True)

mu.setVal(3)
# Add the morphing function to the workspace
w.Import(morph)

# Generate data from the gaussian distribution with mean value 1 
data = w["g2"].generate(ROOT.RooArgSet(x), 1000)  # Adjust the number of data points as needed

# Create the NLL from the generated data using the morphed function
nll = w["morph"].createNLL(data) 
input("")
# plot
c1 = ROOT.TCanvas()
frame = mu.frame()
nll.plotOn(frame, ShiftToZero=True)
frame.Draw()

# Minimize the NLL
minimizer = ROOT.RooMinimizer(nll)
minimizer.minimize('Minuit2')
result = minimizer.save()
result.Print()