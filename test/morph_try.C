#include "RooRealVar.h"
#include "RooRealVar.h"
#include "RooWorkspace.h"
#include "RooGaussian.h"
#include "RooUniform.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooMomentMorphFuncND.h"

using namespace RooFit;

// Number of samples to fill the histograms
const int n_samples = 100;

// Define the morphing routine
void morphing(RooWorkspace *ws, int setting)
{
    // Get Variables from the workspace
    RooRealVar *x_var = ws->var("x");
    RooRealVar *mu_var = ws->var("mu");
    RooRealVar *sigma_var = ws->var("sigma");
    RooAbsPdf *gauss = ws->pdf("gauss");

    // Initialize a plot
    RooPlot *frame1 = x_var->frame();

    // Define binning for morphing
    RooMomentMorphFuncND::Grid2 grid(RooBinning(4, 0.0, 4.0));

    // Set binning of histograms, has to be customized for optimal results
    x_var->setBins(50);

    std::vector<int> parampoints = {0, 1, 2, 3, 4};

    for (auto i : parampoints)
    {
        // Define the sampled gaussians
        RooRealVar mu_help(Form("mu%d", i), Form("mu%d", i), i);
        // Use * beacause RooGuassian expexts objects no pointers
        RooGaussian help(Form("g%d", i), Form("g%d", i), *x_var, mu_help, *sigma_var);
        ws->import(help);

        // Fill the histograms
        // unique pointer to prevent memory leaks
        // RooAbsPdf *help = ws->pdf(Form("g%d", i));

        std::unique_ptr<RooDataHist> hist1{dynamic_cast<RooDataHist*>(ws->pdf(Form("g%d", i))->generateBinned(*x_var, n_samples))};
        // std::unique_ptr<RooDataHist> hist1{*ws->pdf(Form("g%d", i))->generateBinned(*x_var, n_samples)};
        ws->import(RooHistPdf(Form("histpdf%d", i), Form("histpdf%d", i), *x_var, *hist1, 1));

        RooAbsPdf *pdf = ws->pdf(Form("histpdf%d", i));
        pdf->plotOn(frame1);
        grid.addPdf(*pdf, i);
    }

    // Create the morphing
    RooMomentMorphFuncND morph_func("morpf_func", "morph_func", RooArgList(*mu_var), RooArgList(*x_var), grid, setting);

    // Normalizing the morphed object to be a pdf, set it false to prevent warning messages and gain computational speed up
    //  morph_func-> setPdfMode()

    // Creating the morphed pdf
    RooWrapperPdf morph("morph", "morph", morph_func, true);
    ws-> import(morph);
    morph_ = ws-> pdf("morph");
    morph_->PlotOn(frame1); 


    TCanvas* c0 = new TCanvas();
    frame1->Draw();
}

// Define the workspace
RooWorkspace *build_ws(double mu_observed)
{
    // Generate the 'observed' data
    RooRealVar x_var("x", "x", -5, 15);
    RooRealVar mu_var("mu", "mu", mu_observed, 0, 4);
    RooRealVar sigma_var("sigma", "sigma", 1.5);

    RooGaussian gauss("gauss", "gauss", x_var, mu_var, sigma_var);
    RooUniform uniform("uniform", "uniform", x_var);

    RooDataSet *obs_data = gauss.generate(x_var, n_samples);
    obs_data->SetName("obs_data");

    // Using a workspace for easier processing
    RooWorkspace *ws = new RooWorkspace();
    ws->import(x_var);
    ws->import(mu_var);
    ws->import(sigma_var);
    ws->import(gauss);
    ws->import(uniform);
    ws->import(*obs_data);

    return ws;
}

// Do the example
void morphing_example()
{
    // Define the 'observed' mu
    double mu_observed = 2.5;

    // Import variables from workspace
    RooWorkspace *ws = build_ws(mu_observed);
    RooRealVar *x_var = ws->var("x");
    RooRealVar *mu_var = ws->var("mu");
    RooRealVar *sigma_var = ws->var("sigma");
    RooGaussian *gauss = ws->pdf("gauss");
    RooUniform *uniform = ws->pdf("uniform");
    RooDataSet *obs_data = ws->data("obs_data");

    morphing(ws, RooMomentMorphFuncND::Linear);
}
