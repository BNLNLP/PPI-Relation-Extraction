import os
from argparse import ArgumentParser

data_converter_dict = {"ChemProt_BLURB": "datasets/conversion/convert_chemprot_blurb.py",
                       "DDI_BLURB": "datasets/conversion/convert_ddi_blurb.py",
                       "GAD_BLURB": "datasets/conversion/convert_gad_blurb.py",
                       "EU-ADR_BioBERT": "datasets/conversion/convert_euadr_gad_biobert.py",
                       "PPI": "datasets/conversion/convert_ppi.py",
                       "GRR": "datasets/conversion/convert_grr.py"}

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset to be pre-processed.")
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    
    if dataset_name in ["AImed", "BioInfer", "HPRD50", "IEPA", "LLL", 
                        "AImed_typed", "BioInfer_typed", "HPRD50_typed", "IEPA_typed", "LLL_typed", "BioCreative_type", 
                        "BioCreative" ''' not used for now''']: 						
        dataset_name = "PPI"
    elif dataset_name in ["P-putida"]: 						
        dataset_name = "GRR"
    
    f = os.path.join(os.getcwd(), data_converter_dict[dataset_name])
    exec(open(f).read(), {"dataset_name": args.dataset_name})
    
    
if __name__ == "__main__":
    main()	
