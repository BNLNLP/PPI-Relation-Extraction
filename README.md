# PPI-Relation-Extraction
This is the official code of the paper

Protein-Protein Interactions (PPIs) Extraction from Biomedical Literature using Attention-based Relational Context Information (link will be added.)

## Overview
The project aims to build a Protein-Protein Interaction (PPI) extraction model based on Transformer architecture. 
We used the five PPI benchmark datasets and three relation extraction (RE) datasets to evaluate our model.
We provide the extended version of PPI datasets, called typed PPI, which have further augmented those positive/negative calls with own PPI role labels (structural or enzymatic). <br/>

![PPI_RE_architecture](img/model_architecture.jpg)

## Installation
You may install the dependencies via either conda or pip. Generally, NBFNet works with Python 3.7/3.8 and PyTorch version >= 1.8.0.
Python version is 3.7, and the versions of needed packages are listed in requirements.txt

## Biomedical Relation Extraction benchmark data
* ChemProt (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6051439/)
* DDI (https://www.sciencedirect.com/science/article/pii/S1532046413001123?via%3Dihub)
* GAD (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0472-9)
* EU-ADR (https://www.sciencedirect.com/science/article/pii/S1532046412000573)

## PPI benchmark data
* AIMed (https://www.sciencedirect.com/science/article/pii/S0933365704001319)
* BioInfer (https://link.springer.com/article/10.1186/1471-2105-8-50)
* HPRD50 (https://academic.oup.com/bioinformatics/article/23/3/365/236564)
* IEPA (http://psb.stanford.edu/psb-online/proceedings/psb02/ding.pdf)
* LLL (https://hal.inrae.fr/hal-02762818/document)

## Typed PPI data
The data annotation is based on the five PPI benchmark data above plus BioCreative VI.
* BioCreative VI (Track 4: Mining protein interactions and mutations for precision medicine (PM))

## Reproduction

## Results

### Biomedical Relation Extraction benchmark data ###
   
<table>
    <tr>
        <th>Method</th>
        <th>ChemProt</th>
        <th>DDI</th>
        <th>GAD</th>
        <th>EU-ADR</th>
    </tr>
	<tr>
        <th>SOTA</th>
        <td>77.5</td>
        <td>83.6</td>
        <td>84.3</td>
        <td>85.1</td>
    </tr>
    <tr>
        <th>Ours (Entity Mention Pooling + Relation Context)</th>
        <td>80.1</td>
        <td>81.3</td>
        <td>85.0</td>
        <td>86.0</td>
    </tr>
    <tr>
        <th>Ours (Entity Start Marker + Relation Context)</th>
        <td>79.2</td>
        <td>83.6</td>
        <td>84.5</td>
        <td>85.5</td>
    </tr>
</table>

### PPI benchmark data ###
     
<table>
    <tr>
        <th>Method</th>
        <th>AIMed</th>
        <th>BioInfer</th>
        <th>HPRD50</th>
        <th>IEPA</th>
		<th>LLL</th>
		<th>Avg.</th>
    </tr>
	<tr>
        <th>SOTA</th>
        <td>83.9</td>
        <td>90.3</td>
        <td>85.5</td>
        <td>84.9</td>
		<td>89.2</td>
        <td>86.5</td>
    </tr>
    <tr>
        <th>Ours (Entity Mention Pooling + Relation Context)</th>
        <td>90.8</td>
        <td>88.2</td>
        <td>84.5</td>
        <td>85.9</td>
		<td>84.6</td>
        <td>86.8</td>
    </tr>
    <tr>
        <th>Ours (Entity Start Marker + Relation Context)</th>
        <td>92.0</td>
        <td>91.3</td>
        <td>88.2</td>
        <td>87.4</td>
		<td>89.4</td>
        <td>89.7</td>
    </tr>
</table>

### Typed PPI data ###

<table>
    <tr>
        <th>Method</th>
        <th>Typed PPI</th>
    </tr>
    <tr>
        <th>Ours (Entity Mention Pooling + Relation Context)</th>
        <td>86.4</td>
    </tr>
    <tr>
        <th>Ours (Entity Start Marker + Relation Context)</th>
        <td>87.8</td>
    </tr>
</table>


## Citation
TBA

<!-- reference from NBFNet
```bibtex
@article{zhu2021neural,
  title={Neural bellman-ford networks: A general graph neural network framework for link prediction},
  author={Zhu, Zhaocheng and Zhang, Zuobai and Xhonneux, Louis-Pascal and Tang, Jian},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
-->

<!--
### Prerequisites
Install the following packages.

* HuggingFace Transformers (https://github.com/huggingface/transformers)
* Scikit-learn (https://scikit-learn.org)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* This work has been authored by employees of Brookhaven Science Associates, LLC operated under Contract No. DESC0012704. The authors gratefully acknowledge the funding support from the Brookhaven National Laboratory under the Laboratory Directed Research and Development 18-05 FY 18-20.
-->