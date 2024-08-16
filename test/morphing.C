#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooUniform.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooWorkspace.h"
#include "RooMinimizer.h"
#include "RooMomentMorphFuncND.h"
#include "RooHistPdf.h"
#include "RooFormulaVar.h"
#include "RooMomentMorphFuncND.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace RooFit;

// Number of samples used to fill the histograms, adjust it to your needs
const int n_samples = 100;

// Function to perform morphing
void morphing(RooWorkspace* workspace, int setting) {
    // Get variables from the workspace
    RooRealVar* x_var = workspace->var("x");
    RooRealVar* mu_var = workspace->var("mu");
    RooRealVar* sigma_var = workspace->var("sigma");

    // Set up a frame for plotting
    RooPlot* frame1 = x_var->frame();

    // Define binning for morphing
    RooBinning bin_mu_x(4, 0.0, 4.0);
    RooMomentMorphFuncND::Grid2 grid(bin_mu_x);

    x_var->setBins(50);

    // Number of 'sampled' gaussians, if you change it, adjust the binning properly
    std::vector<int> parampoints = {0, 1, 2, 3, 4};

    for (auto i : parampoints) {
        // Define the sampled Gaussians
        RooRealVar mu_help(Form("mu%d", i), Form("mu%d", i), i);
        RooGaussian help(Form("g%d", i), Form("g%d", i), *x_var, mu_help, *sigma_var);
        workspace->import(help);

        // Fill the histograms
        RooDataHist* hist = workspace->pdf(Form("g%d", i))->generateBinned(RooArgSet(*x_var), n_samples);

        // Make sure that every bin is filled and we don't get zero probability
        for (int i_bin = 0; i_bin < hist->numEntries(); ++i_bin) {
            hist->add(hist->get(i_bin), 1.0);
        }

        // Add the PDF to the workspace, the intOrder of 1 is necessary for calculation of the NLL
        workspace->import(RooHistPdf(Form("histpdf%d", i), Form("histpdf%d", i), RooArgSet(*x_var), *hist, 1));

        // Add the PDF to the grid
        grid.addPdf(*workspace->pdf(Form("histpdf%d", i)), i);

        // For plotting purposes
        workspace->pdf(Form("histpdf%d", i))->plotOn(frame1);
    }

    // Create the morphing and add it to the workspace
    RooMomentMorphFuncND morph_func("morph_func", "morph_func", RooArgList(*mu_var), RooArgList(*x_var), grid, setting);

    // Normalize the morphed object to be a PDF, set it false to prevent warning messages and computational speed up
    morph_func.setPdfMode(false);

    // Creating the morphed PDF
    RooWrapperPdf morph("morph", "morph", morph_func, true);
    workspace->import(morph);
    workspace->pdf("morph")->plotOn(frame1, LineColor(kRed));

    TCanvas* c0 = new TCanvas();
    frame1->Draw();

    // Prevent the plot from being closed instantly
    std::cin.get();
}

// Function to define the "observed" data in a workspace
RooWorkspace* build_ws(double mu_observed) {
    RooRealVar x_var("x", "x", -5, 15);
    RooRealVar mu_var("mu", "mu", mu_observed, 0, 4);
    RooRealVar sigma_var("sigma", "sigma", 1.5);

    RooGaussian gauss("gauss", "gauss", x_var, mu_var, sigma_var);
    RooUniform uniform("uniform", "uniform", x_var);

    RooDataSet* obs_data = gauss.generate(x_var, n_samples);
    obs_data->SetName("obs_data");

    // Using a workspace for easier processing inside the class
    RooWorkspace* workspace = new RooWorkspace();
    workspace->import(x_var);
    workspace->import(mu_var);
    workspace->import(sigma_var);
    workspace->import(gauss);
    workspace->import(uniform);
    workspace->import(*obs_data);

    return workspace;
}

void morphing_example() {
    // The "observed" data
    double mu_observed = 2.5;
    RooWorkspace* workspace = build_ws(mu_observed);

    RooRealVar* mu_var = workspace->var("mu");
    RooRealVar* x_var = workspace->var("x");
    RooRealVar* sigma_var = workspace->var("sigma");
    RooAbsPdf* gauss = workspace->pdf("gauss");
    RooAbsPdf* uniform = workspace->pdf("uniform");
    RooDataSet* obs_data = (RooDataSet*)workspace->data("obs_data");

    // Compute the real likelihood ratio
    RooFormulaVar llhr_calc("llhr_calc", "x[0] / (x[0] + x[1])", RooArgList(*gauss, *uniform));

    // Create the exact negative log likelihood functions for Gaussian model
    RooNLLVar* nll_gauss = gauss->createNLL(*obs_data);

    // Compute the morphed NLL
    morphing(workspace, RooMomentMorphFuncND::Linear);
    RooNLLVar* nll_morph = workspace->pdf("morph")->createNLL(*obs_data);

    // Plot the negative logarithmic summed likelihood
    TCanvas* c1 = new TCanvas();
    RooPlot* frame = mu_var->frame(Title("SBI vs. Morphing"));
    nll_gauss->plotOn(frame, LineColor(kBlue), ShiftToZero());
    nll_morph->plotOn(frame, LineColor(kRed), ShiftToZero());
    frame->Draw();

    // Create a legend and add entries
    TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9);  // Adjust coordinates as needed
    legend->AddEntry("gauss", "Gaussian", "l");
    legend->AddEntry("morph", "Morphed", "l");
    legend->Draw();

    // Compute the minimum via Minuit and display the results
    std::vector<RooNLLVar*> nlls = {nll_gauss, nll_morph};
    for (auto nll : nlls) {
        RooMinimizer minimizer(*nll);
        minimizer.setErrorLevel(0.5);  // Adjust the error level in the minimization to work with likelihoods
        minimizer.setPrintLevel(-1);
        minimizer.minimize("Minuit2");
        RooFitResult* result = minimizer.save();
        result->Print();
    }
}
