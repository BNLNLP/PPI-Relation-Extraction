# PPI-Relation-Extraction
The project aims to build a Protein-Protein Interaction (PPI) extraction model based on Transformer architecture. 
We used the five PPI benchmark datasets and three relation extraction (RE) datasets to evaluate our model.
We provide the extended version of PPI datasets, called typed PPI, which have further augmented those positive/negative calls with own PPI role labels (structural or enzymatic). <br/>
** The model specifications and the typed-PPI data will be released after a paper in preparation is published. **


## PPI benchmark data
* AIMed (https://www.sciencedirect.com/science/article/pii/S0933365704001319)
* BioInfer (https://link.springer.com/article/10.1186/1471-2105-8-50)
* HPRD50 (https://academic.oup.com/bioinformatics/article/23/3/365/236564)
* IEPA (http://psb.stanford.edu/psb-online/proceedings/psb02/ding.pdf)
* LLL (https://hal.inrae.fr/hal-02762818/document)

## Typed PPI data
The data annotation is based on the five benchmark data above plus BioCreative VI.
* BioCreative VI (Track 4: Mining protein interactions and mutations for precision medicine (PM))

## RE benchmark data
* ChemProt (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6051439/)
* GAD (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0472-9)
* EU-ADR (https://www.sciencedirect.com/science/article/pii/S1532046412000573)

### Prerequisites
Install the following packages.

* HuggingFace Transformers (https://github.com/huggingface/transformers)
* Scikit-learn (https://scikit-learn.org)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

<!--
## Acknowledgments
* This work has been authored by employees of Brookhaven Science Associates, LLC operated under Contract No. DESC0012704. The authors gratefully acknowledge the funding support from the Brookhaven National Laboratory under the Laboratory Directed Research and Development 18-05 FY 18-20.
-->