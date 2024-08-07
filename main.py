import pandas as pd
from src.data.data_loader import load_data
from src.data.data_preprocessing import preprocess_data, fetch_smiles
from src.models.algorithm import extract_features, evaluate_with_lazy_regressor, feature_selection_and_optimization
from transformers import AutoTokenizer, RobertaModel, BertModel
import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description="Choose model for feature extraction")
    parser.add_argument('--model', type=str, choices=['kbert', 'chemberta'], default='chemberta',
                        help="Model to use for feature extraction")
    args = parser.parse_args()

    input_file_path = 'path/to/compound_data.xlsx'

    df = load_data(input_file_path)
    df = preprocess_data(df)
    compound_names = df.iloc[:, :2].values.tolist()
    smiles_list = fetch_smiles(compound_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'kbert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        pretrained_model_path = 'path/to/pretrain_k_bert_epoch_7.pth'
        model = BertModel.from_pretrained('bert-base-uncased')
        pretrained_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(pretrained_dict, strict=False)
    elif args.model == 'chemberta':
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        model = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

    model.to(device)
    model.eval()

    smiles1 = df['SMILES1'].tolist()
    smiles2 = df['SMILES2'].tolist()
    features_smiles1 = extract_features(smiles1, model, tokenizer, device)
    features_smiles2 = extract_features(smiles2, model, tokenizer, device)
    descriptors = df.drop(columns=['SMILES1', 'SMILES2', 'DES melting temperature_K']).values
    features = np.hstack([features_smiles1, features_smiles2, descriptors])
    melting_temps = df['DES melting temperature_K'].values

    evaluate_with_lazy_regressor(features, melting_temps)
    feature_selection_and_optimization(features, melting_temps)


if __name__ == "__main__":
    main()
