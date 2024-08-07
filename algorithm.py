import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from transformers import BertModel, RobertaModel, AutoTokenizer
from lazypredict.Supervised import LazyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        inputs = self.tokenizer(smiles, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'target': torch.tensor(target, dtype=torch.float)}


class ChemBERTaRegressor(nn.Module):
    def __init__(self, chemberta_model):
        super(ChemBERTaRegressor, self).__init__()
        self.chemberta_model = chemberta_model
        self.regressor = nn.Linear(chemberta_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_output)
        return prediction.squeeze()


def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        optimizer.zero_grad()
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def extract_features(smiles_list, model, tokenizer, device):
    model.eval()
    features = []
    for smiles in smiles_list:
        inputs = tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        features.append(cls_output.cpu().numpy())
    return np.concatenate(features, axis=0)


def evaluate_with_lazy_regressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(models)


def feature_selection_and_optimization(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_

    n = 10
    indices = np.argsort(importances)[-n:]
    selected_features = X[:, indices]

    X_selected_scaled = scaler.fit_transform(selected_features)
    rf.fit(X_selected_scaled, y)
    rf_predictions = rf.predict(X_selected_scaled).reshape(-1, 1)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_selected_scaled)
    initial_centroids = kmeans.cluster_centers_

    kmeans = KMeans(n_clusters=5, init=initial_centroids, n_init=1, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_selected_scaled)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, partial_index in sss.split(X, kmeans_labels):
        partial_labels = kmeans_labels[partial_index]

    partial_indices = partial_index
    kmeans_labels[partial_indices] = partial_labels

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(X_selected_scaled)
    X_selected_scaled = np.hstack([X_selected_scaled, pca_features])

    X_train, X_test, y_train, y_test = train_test_split(X_selected_scaled, y, test_size=0.2, random_state=42)

    et_model = ExtraTreesRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', 0.5, 0.7],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=et_model, param_grid=param_grid, cv=5, n_jobs=-1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_et_model = grid_search.best_estimator_
    y_pred = best_et_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
