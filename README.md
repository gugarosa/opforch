# OPForch: A PyTorch-Inspired Optimum-Path Forest Classifier

[![Latest release](https://img.shields.io/github/release/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/issues)
[![License](https://img.shields.io/github/license/gugarosa/opforch.svg)](https://github.com/gugarosa/opforch/blob/master/LICENSE)

## Welcome to OPForch.

*Note that this implementation relies purely on the standard [LibOPF](https://github.com/jppbsi/LibOPF). Therefore, if one uses our package, please also cite the original LibOPF [authors](https://github.com/jppbsi/LibOPF/wiki/Additional-information).*

Tired of traditional classifiers? In search for a novel graph-based classifier? Wants to classify data using CUDA? If yes, OPForch is for you! This package is an innovative way of dealing with an Optimum-Path Forest classifier. Builded from the ground using a PyTorch-only approach, we are here to reduce memory footprint and foster every fast and malleable computation.

Use OPForch if you need a library or wish to:

* Create your datasets;
* Design or use pre-loaded state-of-art classifiers;
* Mix-and-match different strategies to solve your problem;
* Because it is cool to classify things.

Read the docs at [opforch.readthedocs.io](https://opforch.readthedocs.io).

OPForch is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**. Call us.

---

## Citation

If you use OPForch to fulfill any of your needs, please cite us:

---

## Datasets

In search for datasets? We have some already pre-loaded into OPF file format. Just check them out at our [website](http://recogna.tech)!

---

## Getting started: 60 seconds with OPForch

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think.

Alternatively, if you wish to learn even more, please take a minute:

OPForch is based on the following structure, and you should pay attention to its tree:

```yaml
- opforch
    - core
        - heap
        - node
    - utils
        - constants
        - logging
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use it as you wish than re-implementing the same thing over and over again.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, OPForch will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```bash
pip install opforch
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```bash
pip install -e .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
