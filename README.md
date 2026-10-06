# NorthStar

### Navigating chemical space through reference-guided molecular generation

NorthStar is a computational research pipeline that connects molecular information encoded as **SMILES** with deep generative learning and structure-based assessment. A **conditional variational autoencoder (CVAE)** learns molecular representations and generates candidate structures conditioned on a chosen physicochemical property profile. Molecular docking and consensus pharmacophore analysis form the downstream assessment workflow.

Developed as part of doctoral research, NorthStar explores a central question:

> Can a known drug act as a compass for discovering new molecular structures with a desired property profile?

![NorthStar workflow: molecular knowledge, learned chemical space, reference-guided exploration, and structural assessment.](docs/images/northstar-workflow.svg)

*Conceptual overview. Molecular structures, latent-space constellations, and binding-site illustrations are schematic.*

## The idea

Public chemical databases contain extensive molecular information in compact, machine-readable forms. SMILES represents molecular connectivity as a string, making it possible to learn from chemical structures using sequence models.

NorthStar pairs these strings with physicochemical descriptors and trains a CVAE to learn a continuous latent representation of molecular space. A reference drug—the **North Star**—provides the property profile used to condition generation.

The aim is to explore new structures with reference-like properties. Generated structures are computational candidates: their biological activity, safety, and suitability for development require further evaluation.

## Research workflow

### 1. Represent molecular knowledge

Chemical structures are represented as SMILES and paired with calculated descriptors:

- **Molecular weight:** exact molecular weight calculated with RDKit.
- **Lipophilicity:** calculated logP.
- **Polarity:** topological polar surface area (TPSA).

The research workflow combines the **ZINC 250K dataset** with an **in-house dataset of viral polymerase inhibitors** assembled from public molecular databases.

### 2. Learn the chemical landscape

A conditional variational autoencoder learns from molecular strings and their properties. Its LSTM encoder maps the input to a latent distribution, and its LSTM decoder reconstructs molecular sequences using the latent representation and property conditions.

Training combines a sequence reconstruction objective with a Kullback–Leibler regularization term. Together, these objectives support a continuous latent space from which molecular sequences can be generated.

### 3. Generate candidates using a reference profile

The generation script samples latent vectors from a normal distribution and supplies a target property vector to the decoder. The reference profile guides generation through **property conditioning**; the published script does not explicitly sample around the encoded coordinates of a reference molecule.

Generated sequences are converted to SMILES, duplicate strings are removed, and RDKit is used to retain parseable molecular structures. The output includes SMILES and recalculated molecular descriptors.

### 4. Assess molecular interactions

The broader research workflow evaluates generated candidates through **molecular docking** in the target enzyme’s binding site and **consensus pharmacophore analysis**. These steps support the investigation of binding poses and shared interaction features for candidate prioritization.

Docking and pharmacophore analysis are shown in the research workflow; their execution scripts are not currently included in this repository.

## The North Star: Remdesivir

In the presented research, **Remdesivir** serves as the reference drug for an antiviral application focused on **RNA-dependent RNA polymerase (RdRp)**.

The reference provides a physicochemical profile for exploring alternative molecular structures. The resulting candidates can be viewed as *constellations*: new molecular possibilities organized around a guiding property profile.

Similarity in these properties does not, by itself, establish equivalent target binding or biological activity. Structural diversity and novelty also need to be measured independently.

## Model architecture

The implementation uses recurrent sequence models for both encoding and decoding.

| Component | Configuration in the supplied scripts |
| --- | --- |
| Generative model | Conditional variational autoencoder |
| Encoder | 3 LSTM layers, 512 units per layer |
| Decoder | 3 LSTM layers, 512 units per layer |
| Latent representation | 200 dimensions |
| Property conditions | Molecular weight, logP, TPSA |
| Maximum sequence length | 120 tokens, including sequence handling |
| Training objective | Reconstruction loss + KL regularization |
| Optimizer | Adam |

*Values reflect the default configuration. Some settings are exposed through command-line arguments; the decoder state initialization currently assumes three layers.*

![CVAE architecture showing the LSTM encoder, latent space, and LSTM decoder.](https://github.com/user-attachments/assets/0411341a-5e33-4b3e-8c0c-e0b1ac4e6e4f)


## Structural assessment illustration

The original research overview presents two candidate molecules positioned within the target enzyme’s binding pocket.

![Two candidate molecules shown within the target enzyme binding pocket.](https://github.com/user-attachments/assets/9892dadf-899a-4f0a-8be1-4ffae0685ee6)

*Illustration of the structure-based assessment stage. Binding poses alone do not establish experimental activity.*

## Repository guide

| File | Purpose |
| --- | --- |
| `core_architecture.py` | CVAE definition, encoder, decoder, losses, and sampling methods |
| `core_train.py` | Training loop, held-out loss evaluation, and checkpoint saving |
| `constellating.py` | Property-conditioned generation and candidate export |
| `prop_calc_module_basis.py` | RDKit descriptor calculation from SMILES |
| `utils.py` | Sequence preparation and molecular representation utilities |
| `gpu-test.py` | GPU diagnostic script |
| `test.py` | Data preparation utility that extracts alternating input lines |

## Running and reproducing the work

This repository currently contains a **research implementation** of the molecular generation component. A fully specified, end-to-end reproduction environment is not yet included.

### Software components

The scripts use Python, TensorFlow with TensorFlow 1.x compatibility APIs, TensorFlow Addons, RDKit, NumPy, and h5py. A tested dependency lockfile is not supplied, so compatible versions must be established before running the model.

### Input and output formats

The descriptor preparation script accepts a text file containing one SMILES string per line. With RDKit installed, its command-line interface is:

```bash
python prop_calc_module_basis.py \
  --input_filename smiles.txt \
  --output_filename smiles_prop.txt
```

The resulting training-data file contains tab-separated values in this order, without a header:

```text
SMILES    exact_molecular_weight    logP    TPSA
```

Candidate generation requires a compatible trained checkpoint, the vocabulary derived from the property dataset, and target properties in the same order. Its exported table contains `smiles`, `MW`, `LogP`, and `TPSA` columns.

### Current reproduction requirements

Before training and generation can be run as documented entry points, the following naming inconsistencies need to be resolved:

- Training and generation import `CVAE` from `model`, while its definition is supplied in `core_architecture.py`.
- Both scripts call `load_data`, while the corresponding preparation function in `utils.py` is named `load_datasource`.
- Generation calls `convert_to_smiles`, while the available conversion helper is named `conv_to_smiles`.

The repository also does not currently include the training datasets, pretrained checkpoints, or the downstream docking and pharmacophore protocols. These materials and a verified environment are needed for full reproduction.

## Scope and interpretation

NorthStar supports **computational hypothesis generation and candidate prioritization**. Molecular generation does not imply chemical synthesis, and RDKit parsing checks do not establish synthetic accessibility. Shared physicochemical properties do not guarantee shared pharmacological effects.

The conceptual workflow includes downstream structural assessment, while the published Python implementation focuses on data preparation, CVAE training, and molecular generation. Quantitative claims about novelty, diversity, binding performance, or experimental efficacy should be supported by the corresponding evaluation data.

## Project and contact

NorthStar was developed as part of doctoral research. For questions about the research, datasets, or associated publications, contact the maintainer through the [GitHub repository](https://github.com/IoDiakou/NorthStar).
