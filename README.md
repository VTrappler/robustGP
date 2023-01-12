# robustGP

**WORK IN PROGRESS**

## Example

See the [demo notebook](https://github.com/VTrappler/robustGP/blob/dev/notebooks/demo.ipynb) for an example of usage

## Overview

This package has been implemented in order to be able to test and prototype sequential methods based on adaptive designs and Gaussian Processes.

The name may soon be changed to something like `Adaptivemethods` or `SURpy`.

It relies on

- `numpy` for array manipulation
- `scipy` for optimization
- `scikit-learn` for the creation and manipulation of Gaussian Processes

Other backend can be implemented, especially for GPs

## Usage

The class `AdaptiveStrategy` in the file SURmodel.py is the main interface for runnning this kind of experiments, and is used to define the methods related to GP (fit, add points to design, evaluate true underlying function etc)

To run an `AdaptiveStrategy`, one needs first to define the corresponding `Enrichment`.

This `Enrichment` may take several forms

- OneStep enrichment, which are based on `Acquisition` functions which are optimized to select the new points
- Sampling based enrichment, (`AKMCSEnrichment`)
