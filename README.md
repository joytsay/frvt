Python code to test NIST frvt accuracy of MUGSHOT &amp; pnas datasets

```
cd 11
./compileAndRunFrvt.sh
pip install -r requirements.txt
python plot.py
```

* NIST comes with the MUGSHOT dataset for testing:
  * 653 pairs (1306 img ppm files)
* pnas is also used for quick verification:
  * 20 pairs (12 Genuine(G) pairs & 8 Imposter(I) pairs)
* Additional tensorflow FR model & lib/*.so need to be downloaded seperately:
  * 09-02_02-45.pb

## G-I Similarity box scatter chart

* MUGSHOT:

![Alt text](11/GIboxPlot.png?raw=true "Title")

* pnas:

![Alt text](11/GIboxPlotPNAS.png?raw=true "Title")


# Face Recognition Vendor Test (FRVT) Validation Packages
This repository contains validation packages for all [Ongoing FRVT evaluation](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt-ongoing) tracks.
We recommend developers clone the entire repository and run validation from within
the folder that corresponds to the evaluation of interest.  The ./common directory
contains files that are shared across all validation packages.

