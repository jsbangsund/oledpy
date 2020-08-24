# oledpy
Package for calculating common parameters for thin film light-emitting devices (LEDs), including:
 1) light outcoupling efficiency based on a classical dipole emission model
 2) electric field profile within device, for optical pumping experiments
 3) Efficiency, color coordinates, and other common metrics

 Note: outcoupling calculations are (currently) only valid for bottom-emitting geometries, where light emission occurs through the substrate. I have checked these calculations with literature reports and the commercial software, [SETFOS](https://www.fluxim.com/setfos-intro), and find that the results reproduce all spectral features and trends (with thickness, index, etc.). Absolute comparisons of outcoupling efficiency generally agree within 1-2%.

 ## Power dissipation spectra:

 <img src="https://github.com/jsbangsund/oledpy/blob/master/Plots/power_dissipation_2d.png" height="250">

 ## Calculating mode distributions with just a few lines of code:

 <img src="https://github.com/jsbangsund/oledpy/blob/master/Plots/notebook_example.PNG" width="500">

# Installation
I recommend using Anaconda to create an environment for this package, using the included oled.yml environment file. If you don't have Anaconda, I recommend [installing Miniconda](https://docs.conda.io/en/latest/miniconda.html).

To build from an environment file (note that the name of the env is defined in the first line of oled.yml):
```
conda env create -f oled.yml
```

Once this environment is built, you can activate the environment using:
```
conda activate oled
```

Then you can install the oledpy package using `python setup.py install`. Make sure that you are in the directory of the package when you run this.

Another method is to call this command from the directory of the package:
```bash
pip install -e .
```
This uses a symlink, so that changes to the source files on your computer (either by manual edits or updating from github) will be immediately available within the package. This is convenient, as you won't need to re-install the package every time something is changed.


# References:
 1. Furno, M.; Meerheim, R.; Hofmann, S.; Lüssem, B.; Leo, K.
      Efficiency and Rate of Spontaneous Emission in Organic Electroluminescent Devices.
      Phys. Rev. B 2012, 85 (11), 115205.

 2. Furno, M.; Meerheim, R.; Thomschke, M.; Hofmann, S.; Lüssem, B.; Leo, K.
      Outcoupling Efficiency in Small-Molecule OLEDs: From Theory to Experiment.
      In Proc. SPIE; 2010; Vol. 7617, p 761716.

 3. Neyts, K. A. Simulation of Light Emission from Thin-Film Microcavities.
      J. Opt. Soc. Am. A, JOSAA 1998, 15 (4), 962–971.

 4. Byrnes, Steven. Multilayer Optical Calculations, 2016.
      https://arxiv.org/abs/1603.02720, https://github.com/sbyrnes321/tmm

 5. Pettersson, Leif AA, Lucimara S. Roman, and Olle Inganäs.
      Modeling photocurrent action spectra of photovoltaic devices based on organic thin films.
      Journal of Applied Physics 86.1 (1999): 487-496.

 6. Kim, J., Kang, K., Kim, K. Y., & Kim, J. (2017).
      Origin of a sharp spectral peak near the critical angle in the spectral power
      density profile of top-emitting organic light-emitting diodes.
      Japanese Journal of Applied Physics, 57(1), 012101.
      https://iopscience.iop.org/article/10.7567/JJAP.57.012101/meta

 7. Salehi, A., Ho, S., Chen, Y., Peng, C., Yersin, H., and So, F.
      Highly Efficient Organic Light-Emitting Diode Using A Low Refractive Index
      Electron Transport Layer. Adv. Optical Mater. 2017, 5, 1700197
      http://dx.doi.org/10.1002/adom.201700197
