# PPI-Relation-Extraction
The project aims to build a Protein-Protein Interaction (PPI) extraction model based on Transformer architecture. 
We used the five PPI benchmark datasets and three relation extraction (RE) datasets to evaluate our model.
We provide the extended version of PPI datasets, called typed PPI, which have further augmented those positive/negative calls with own PPI role labels (structural or enzymatic). <br/>
** The model specifications and the typed-PPI data will be released after a paper in preparation is published. **

![PPI_RE_architecture](img/model_architecture.jpg)


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
        <th>GAD@1</th>
        <th>EU-ADR@3</th>
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
        <th>Dataset</th>
        <th>AUROC</th>
        <th>AP</th>
    </tr>
    <tr>
        <th>Cora</th>
        <td>0.956</td>
        <td>0.962</td>
    </tr>
    <tr>
        <th>CiteSeer</th>
        <td>0.923</td>
        <td>0.936</td>
    </tr>
    <tr>
        <th>PubMed</th>
        <td>0.983</td>
        <td>0.982</td>
    </tr>
</table>

### Typed PPI data ###

<table>
    <tr>
        <th rowspan="2">Dataset</th>
        <th colspan="4">HITS@10 (50 sample)</th>
    </tr>
    <tr>
        <th>v1</th>
        <th>v2</th>
        <th>v3</th>
        <th>v4</th>
    </tr>
    <tr>
        <th>FB15k-237</th>
        <td>0.834</td>
        <td>0.949</td>
        <td>0.951</td>
        <td>0.960</td>
    </tr>
    <tr>
        <th>WN18RR</th>
        <td>0.948</td>
        <td>0.905</td>
        <td>0.893</td>
        <td>0.890</td>
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