"""Inference definitions from ordinal_lightgbm_classification.ipynb.

Keep these definitions compatible with the notebook's saved joblib model.
"""
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from joblib.numpy_pickle import NumpyUnpickler
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.DataStructs import ConvertToNumpyArray

CLASS_LABELS = {0: "Not detected", 1: "Low", 2: "High"}
MODEL_PATH = Path(__file__).resolve().parents[1] / "best_ordinal_lgbm_pava.pkl"

def smiles_to_morgan_fp_with_info(smiles, radius=3, n_bits=2048):
    """
    SMILESからMorgan FingerprintとbitInfoを生成する関数
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILESが不正です: {smiles}")
    
    bitInfo = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bitInfo)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr, bitInfo, mol


class OrdinalLightGBM(BaseEstimator, ClassifierMixin):
    """
    累積二値分類による3クラス順序分類器。

    クラス:
        0 = Not detected
        1 = Low
        2 = High

    学習する二値問題:
        model_above_low_: y > 0
        model_high_:      y > 1
    """

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel().astype(int)

        if not np.all(np.isin(y, [0, 1, 2])):
            raise ValueError(
                f"yは0, 1, 2である必要があります: {np.unique(y)}"
            )

        y_above_low = (y > 0).astype(int)
        y_high = (y > 1).astype(int)

        binary_params = self.params.copy()
        binary_params["objective"] = "binary"

        self.model_above_low_ = LGBMClassifier(**binary_params)
        self.model_high_ = LGBMClassifier(**binary_params)

        self.model_above_low_.fit(X, y_above_low)
        self.model_high_.fit(X, y_high)

        self.classes_ = np.array([0, 1, 2], dtype=int)
        return self

    @staticmethod
    def _enforce_monotonicity(q1, q2):
        """
        q1=P(y>0), q2=P(y>1) に q1 >= q2 の制約を課す。

        違反したサンプルでは、2値を平均して単調性を回復する。
        """
        q1 = np.asarray(q1, dtype=float).copy()
        q2 = np.asarray(q2, dtype=float).copy()

        violation = q2 > q1
        midpoint = (q1[violation] + q2[violation]) / 2.0
        q1[violation] = midpoint
        q2[violation] = midpoint

        return q1, q2

    def predict_cumulative_proba(self, X):
        """P(y>0)とP(y>1)を返す。"""
        X = np.asarray(X, dtype=np.float32)

        q1 = self.model_above_low_.predict_proba(X)[:, 1]
        q2 = self.model_high_.predict_proba(X)[:, 1]

        q1, q2 = self._enforce_monotonicity(q1, q2)
        return np.column_stack([q1, q2])

    def predict_proba(self, X):
        cumulative = self.predict_cumulative_proba(X)
        q1 = cumulative[:, 0]
        q2 = cumulative[:, 1]

        p_low = 1.0 - q1
        p_middle = q1 - q2
        p_high = q2

        proba = np.column_stack([p_low, p_middle, p_high])
        proba = np.clip(proba, 0.0, 1.0)
        proba /= proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class OrdinalPAVACalibrator:
    """3クラス順序分類の累積確率をPAVAで補正する。"""

    def fit(self, cumulative_proba, y):
        cumulative_proba = np.asarray(cumulative_proba, dtype=float)
        y = np.asarray(y).ravel().astype(int)
        targets = np.column_stack([y > 0, y > 1]).astype(int)
        self.calibrators_ = []
        for threshold in range(2):
            calibrator = IsotonicRegression(
                increasing=True, out_of_bounds="clip",
                y_min=0.0, y_max=1.0,
            )
            calibrator.fit(cumulative_proba[:, threshold], targets[:, threshold])
            self.calibrators_.append(calibrator)
        return self

    def transform_cumulative(self, cumulative_proba):
        cumulative_proba = np.asarray(cumulative_proba, dtype=float)
        calibrated = np.column_stack([
            calibrator.predict(cumulative_proba[:, threshold])
            for threshold, calibrator in enumerate(self.calibrators_)
        ])
        q1, q2 = OrdinalLightGBM._enforce_monotonicity(
            calibrated[:, 0], calibrated[:, 1]
        )
        return np.clip(np.column_stack([q1, q2]), 0.0, 1.0)

    def predict_proba(self, cumulative_proba):
        cumulative = self.transform_cumulative(cumulative_proba)
        q1, q2 = cumulative[:, 0], cumulative[:, 1]
        proba = np.column_stack([1.0 - q1, q1 - q2, q2])
        proba = np.clip(proba, 0.0, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)


class PAVACalibratedOrdinalModel:
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator
        self.classes_ = model.classes_

    def predict_cumulative_proba(self, X):
        raw = self.model.predict_cumulative_proba(X)
        return self.calibrator.transform_cumulative(raw)

    def predict_proba(self, X):
        raw = self.model.predict_cumulative_proba(X)
        return self.calibrator.predict_proba(raw)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def extract_positive_class_shap(shap_result):
    """SHAPのバージョン差を吸収して二値分類の陽性クラスを返す。"""
    values = shap_result.values if hasattr(shap_result, "values") else shap_result
    if isinstance(values, list):
        return np.asarray(values[1], dtype=float)

    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[-1] == 2:
        return values[:, :, 1]
    if values.ndim == 3 and values.shape[0] == 2:
        return values[1]
    raise ValueError(f"想定外のSHAP値のshapeです: {values.shape}")


def extract_positive_class_shap(shap_result):
    """SHAPのバージョン差を吸収して二値分類の陽性クラスを返す。"""

    values = (
        shap_result.values
        if hasattr(shap_result, "values")
        else shap_result
    )

    if isinstance(values, list):
        return np.asarray(
            values[1],
            dtype=float,
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim == 2:
        return values

    if values.ndim == 3 and values.shape[-1] == 2:
        return values[:, :, 1]

    if values.ndim == 3 and values.shape[0] == 2:
        return values[1]

    raise ValueError(
        f"想定外のSHAP値のshapeです: {values.shape}"
    )


def _shap_color(value):
    """正の寄与を赤、負の寄与を青、0を薄い灰色にする。"""
    value = float(np.clip(value, -1.0, 1.0))
    if value > 0:
        return (1.0, 1.0 - value, 1.0 - value)
    if value < 0:
        return (1.0 + value, 1.0 + value, 1.0)
    return (0.92, 0.92, 0.92)


def project_shap_to_molecule(
    compound_index,
    shap_values,
    molecules,
    bitinfo_list,
    compound_info,
    output=None,
    raw_ids=None,
    size=(600, 500),
    show=True,
):
    """保存済みbitInfoを使い、fingerprint SHAP値を原子・結合へ投影する。"""
    if not 0 <= compound_index < len(molecules):
        raise IndexError(f"compound_indexが範囲外です: {compound_index}")

    # Drawing dependencies are only needed for molecular visualization.
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.Mol(molecules[compound_index])
    bit_info = bitinfo_list[compound_index]
    shap_row = np.asarray(shap_values[compound_index], dtype=float).ravel()
    if raw_ids is not None and len(raw_ids) != len(shap_row):
        raise ValueError("SHAP列数とselected_raw_idsの要素数が一致しません。")

    atom_contributions = np.zeros(mol.GetNumAtoms(), dtype=float)
    for column_index, shap_value in enumerate(shap_row):
        if shap_value == 0:
            continue
        bit_id = column_index if raw_ids is None else raw_ids[column_index]
        occurrences = bit_info.get(bit_id, [])
        if not occurrences:
            continue

        occurrence_share = shap_value / len(occurrences)
        for center_atom, env_radius in occurrences:
            atom_indices = {center_atom}
            if env_radius > 0:
                bond_indices = Chem.FindAtomEnvironmentOfRadiusN(
                    mol, env_radius, center_atom
                )
                for bond_index in bond_indices:
                    bond = mol.GetBondWithIdx(bond_index)
                    atom_indices.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            atom_share = occurrence_share / len(atom_indices)
            for atom_index in atom_indices:
                atom_contributions[atom_index] += atom_share

    max_abs = np.max(np.abs(atom_contributions))
    scaled = atom_contributions / max_abs if max_abs > 0 else atom_contributions
    atom_colors = {i: _shap_color(v) for i, v in enumerate(scaled)}
    bond_colors = {}
    for bond in mol.GetBonds():
        value = (scaled[bond.GetBeginAtomIdx()] + scaled[bond.GetEndAtomIdx()]) / 2.0
        bond_colors[bond.GetIdx()] = _shap_color(value)

    rdDepictor.Compute2DCoords(mol)
    prepared = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.drawOptions().useBWAtomPalette()
    drawer.DrawMolecule(
        prepared,
        highlightAtoms=list(atom_colors),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(svg, encoding="utf-8")
        print(f"Molecular SHAP projection saved: {output}")
    if show:
        from IPython.display import SVG, display
        row = compound_info.reset_index(drop=True).iloc[compound_index]
        name = row.get("name", row.get("Name", f"index={compound_index}"))
        print(f"Compound: {name}")
        display(SVG(svg))
    return atom_contributions, svg

class _NotebookUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        # Notebook classes were serialized under __main__; do not mutate it.
        if module == "__main__" and name in {
            "OrdinalLightGBM", "OrdinalPAVACalibrator", "PAVACalibratedOrdinalModel"
        }:
            return globals()[name]
        # NumPy 2 pickles use numpy._core; NumPy 1 uses numpy.core.
        # Remap only during loading, without altering global module aliases.
        if np.lib.NumpyVersion(np.__version__) < "2.0.0":
            if module == "numpy._core" or module.startswith("numpy._core."):
                module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_classification_model():
    # joblib 1.5 added ensure_native_byte_order; support the original app env too.
    kwargs = {}
    if "ensure_native_byte_order" in inspect.signature(NumpyUnpickler).parameters:
        kwargs["ensure_native_byte_order"] = True
    with MODEL_PATH.open("rb") as handle:
        return _NotebookUnpickler(str(MODEL_PATH), handle, **kwargs).load()


def classification_features(smiles_list):
    smiles_list = list(smiles_list)
    if not smiles_list:
        raise ValueError("No SMILES to predict.")
    rows, molecules, bitinfos = [], [], []
    for row, smiles in enumerate(smiles_list, start=1):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Data row {row}: SMILES is empty.")
        try:
            fp, info, mol = smiles_to_morgan_fp_with_info(smiles)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Data row {row}: invalid SMILES: {smiles}") from exc
        rows.append(fp)
        molecules.append(mol)
        bitinfos.append(info)
    return np.asarray(rows, dtype=np.float32), molecules, bitinfos


def prediction_frame(model, features):
    probabilities = model.predict_proba(features)
    predictions = np.argmax(probabilities, axis=1)
    return pd.DataFrame({
        "Prediction": predictions,
        "Predicted_label": [CLASS_LABELS[int(value)] for value in predictions],
        "probability_not_detected": probabilities[:, 0],
        "probability_low": probabilities[:, 1],
        "probability_high": probabilities[:, 2],
    })
