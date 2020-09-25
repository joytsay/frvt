Add python code to validate our api implement with FAR & FRR:
[validate.py](11/validate.py)

```
cd 11
run ./compileAndRunFrvt.sh first
```

[SUCCESS] NIST frvt validation for confidence:  0.73 

All count:  653 

Known:  325 

Unknown:  264 

knownToUnknown:  64 

unknownToKnown 0 

FAR:  0.0 %

FRR:  9.80091883614089 %


# Face Recognition Vendor Test (FRVT) Validation Packages
This repository contains validation packages for all [Ongoing FRVT evaluation](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt-ongoing) tracks.
We recommend developers clone the entire repository and run validation from within
the folder that corresponds to the evaluation of interest.  The ./common directory
contains files that are shared across all validation packages.

