import base64
import numpy as np
import pandas as pd
import mordred
import colorsys
from mordred import Calculator, descriptors
from io import BytesIO
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools
from rdkit.Chem import rdFingerprintGenerator, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


# フィンガープリントを生成してフィンガープリントのみを返す
def generate_fingerprint_df(df, romol_column="ROMol", fp_prefix="MorganFP"):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    fingerprints = []
    for mol in df[romol_column]:
        if mol is None:
            fingerprints.append(None)
        else:
            try:
                # フィンガープリント生成（ビットベクトルとして）
                fp = generator.GetFingerprint(mol).GetOnBits()
                fingerprints.append(fp)
            except Exception as e:
                print(f"Error generating fingerprint: {e}")
                fingerprints.append(None)

    # フィンガープリントを展開して各カラムに分解
    fp_array = np.zeros((len(df), 2048), dtype=int)
    for i, bits in enumerate(fingerprints):
        if bits is not None:
            fp_array[i, list(bits)] = 1

    # DataFrameに変換
    fp_df = pd.DataFrame(fp_array, columns=[f"{fp_prefix}_{i}" for i in range(2048)])
    return fp_df

# Bokehのための画像変換
def mol_to_base64(mol):
    """RDKitのMolオブジェクトからBase64エンコードされた画像を生成"""
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=(200, 200))  # 分子画像を生成
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# RDKitのMolオブジェクトを受け取り、記述子を計算してデータフレームに追加
def add_phenol_check(df, mol_col="ROMol"):
    """
    データフレームにRDKitの特定の記述子（fr_Ar_OH, fr_phenol）を追加する関数。
    
    Parameters:
    df (pd.DataFrame): RDKitのROMol列を持つデータフレーム
    mol_col (str): 分子オブジェクトが格納されている列の名前（デフォルトは"ROMol"）
    
    Returns:
    pd.DataFrame: 選択したRDKit記述子が追加されたデータフレーム
    """
    selected_descriptors = ["fr_Ar_OH", "fr_phenol"]

    # 記述子関数を取得
    descriptor_funcs = {name: func for name, func in Descriptors.descList if name in selected_descriptors}
    
    descriptors = []
    for mol in df[mol_col]:
        if mol is not None:
            descriptor_values = {name: func(mol) for name, func in descriptor_funcs.items()}
        else:
            descriptor_values = {name: None for name in selected_descriptors}
        
        descriptors.append(descriptor_values)

    # 記述子データフレームを作成
    df_descriptors = pd.DataFrame(descriptors)
    
    # 元のデータフレームと結合
    df = pd.concat([df, df_descriptors], axis=1)
    return df

def add_descriptor(df, mol_col="ROMol"):
    """
    データフレームにRDKitの特定の記述子（fr_Ar_OH, fr_phenol）を追加する関数。
    
    Parameters:
    df (pd.DataFrame): RDKitのROMol列を持つデータフレーム
    mol_col (str): 分子オブジェクトが格納されている列の名前（デフォルトは"ROMol"）
    
    Returns:
    pd.DataFrame: 選択したRDKit記述子が追加されたデータフレーム
    """
    selected_descriptors = ["MolWt", "MolLogP"]

    # 記述子関数を取得
    descriptor_funcs = {name: func for name, func in Descriptors.descList if name in selected_descriptors}
    
    descriptors = []
    for mol in df[mol_col]:
        if mol is not None:
            descriptor_values = {name: func(mol) for name, func in descriptor_funcs.items()}
        else:
            descriptor_values = {name: None for name in selected_descriptors}
        
        descriptors.append(descriptor_values)

    # 記述子データフレームを作成
    df_descriptors = pd.DataFrame(descriptors)
    
    # 元のデータフレームと結合
    df = pd.concat([df, df_descriptors], axis=1)
    return df

calc = Calculator(descriptors, ignore_3D=False)

def get_lowest_energy_conformer(mol):
    mol = Chem.AddHs(mol)  # 水素を追加
    num_confs = 1  # 生成するコンフォーマーの数
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
    
    # 各コンフォーマーのエネルギーを計算
    energies = [
        AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol), confId=i
        ).CalcEnergy()
        for i in range(mol.GetNumConformers())
    ]
    
    # 最小エネルギーのコンフォーマーを選択
    min_energy_idx = energies.index(min(energies))
    
    # 最小エネルギーコンフォーマーを最適化
    AllChem.UFFOptimizeMolecule(mol, confId=min_energy_idx)
    return mol, min_energy_idx

# 3. 記述子計算のための関数
def calculate_descriptors(mol):
    try:
        mol, min_energy_idx = get_lowest_energy_conformer(mol)
        
        # 特定のコンフォーマーの座標をコピー
        conf = mol.GetConformer(min_energy_idx)
        mol_with_coords = Chem.Mol(mol)
        mol_with_coords.RemoveAllConformers()
        mol_with_coords.AddConformer(conf, assignId=True)
        
        # 記述子を計算
        result = calc(mol_with_coords)
        return result.asdict()  # 記述子を辞書形式で返す
    except Exception as e:
        print(f"Error calculating descriptors for molecule: {e}")
        return {}

def get_indexes(mol):
    """
    分子内のすべてのフェノール骨格の原子インデックスを取得。
    O, next_O, o_1, m_1, p, m_2, o_2 の順でリストを返す。
    """
    phenol_substructure = Chem.MolFromSmarts("[OH]c1ccccc1")  # フェノール構造
    matches = mol.GetSubstructMatches(phenol_substructure)  # すべてのフェノール骨格を取得

    phenol_list = []
    for match in matches:
        O = None
        C = []
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "O":
                O = idx
            else:
                C.append(idx)
        
        if O is None or len(C) != 6:
            continue  # フェノール骨格の形式に合わない場合はスキップ

        ortho, meta, para, next_O = [], [], None, None
        for c_idx in C:
            distance = rdmolops.GetShortestPath(mol, O, c_idx)
            if len(distance) == 2:
                next_O = c_idx
            elif len(distance) == 3:
                ortho.append(c_idx)
            elif len(distance) == 4:
                meta.append(c_idx)
            elif len(distance) == 5:
                para = c_idx

        if len(ortho) != 2 or len(meta) != 2 or para is None:
            continue  # 期待される構造でない場合はスキップ

        # CIPランクでオルト・メタの順序を決定
        ortho.sort(key=lambda i: int(mol.GetAtomWithIdx(i).GetProp("_CIPRank")))
        meta.sort(key=lambda i: int(mol.GetAtomWithIdx(i).GetProp("_CIPRank")))

        phenol_list.append([O, next_O, ortho[0], meta[0], para, meta[1], ortho[1]])

    return phenol_list  # すべてのフェノール骨格のリスト


def get_gasteiger_charges_for_phenol(mol):
    """
    分子内の全フェノール骨格のGasteiger電荷を計算し、最も電荷密度が高い骨格を返す。
    """
    AllChem.ComputeGasteigerCharges(mol)  # Gasteiger電荷を計算

    phenol_indices = get_indexes(mol)
    if not phenol_indices:
        return None  # フェノール骨格がない場合

    best_phenol = None
    max_charge_density = -float("inf")

    for indices in phenol_indices:
        charges = [float(mol.GetAtomWithIdx(idx).GetProp("_GasteigerCharge")) for idx in indices]
        charge_density = sum(abs(q) for q in charges)  # 電荷密度（絶対値合計）

        if charge_density > max_charge_density:
            max_charge_density = charge_density
            best_phenol = charges

    return best_phenol  # 電荷密度が最大のフェノール骨格の電荷リスト

def calculate_o1_charge_for_phenol_derivatives(df):
    """
    フェノール誘導体のGasteiger電荷のうち、o_1 のみを計算し、DataFrameに追加する。
    """
    o1_charges = []
    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            o1_charges.append(None)
            continue

        try:
            charges = get_gasteiger_charges_for_phenol(mol)
            if charges:
                o1_charges.append(charges[2])  # o_1 の電荷を取得
            else:
                o1_charges.append(None)
        except Exception:
            o1_charges.append(None)

    df["o_1"] = o1_charges  # 新しい列として追加
    return df

def visualize_morgan_shap(mol, shap_values, radius=3, n_bits=2048):
    """
    SHAP値に基づき、Morganフィンガープリントの重要な部分を可視化する関数。

    Args:
        mol (rdkit.Chem.Mol): RDKitの分子オブジェクト
        shap_values (numpy.ndarray or list): SHAP値（ビットごとの寄与度）
        radius (int): Morganフィンガープリントの半径（デフォルト: 3）
        n_bits (int): フィンガープリントのビット数（デフォルト: 2048）

    Returns:
        str: SVG形式の分子画像
    """
    

    shap_values = [shap_values[i, 0] for i in range(len(shap_values))]

    # Morganフィンガープリントを計算
    info = {}
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=info)
    
    # SHAP値の辞書を作成
    shap_dict = {i: shap_values[i] for i in range(len(shap_values))}

    # st.write(len(shap_values))
    # st.write(shap_dict)


    # SHAP値を反映した部分構造の色を設定（半径内の隣接原子も含める）
    atom_contributions_plus = {}  # 正の影響
    atom_contributions_minus = {}  # 負の影響

    for bit, value in shap_dict.items():
        if bit in info:
            for (atom_idx, rad) in info[bit]:

                neighbor_atoms = set()
                # 半径 `radius` の環境（結合のインデックスリスト）を取得
                bond_indices = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                # 取得した結合から隣接原子をカウント
                for bond_idx in bond_indices:
                    bond = mol.GetBondWithIdx(bond_idx)
                    begin_atom = bond.GetBeginAtomIdx()
                    end_atom = bond.GetEndAtomIdx()

                    # 中心原子でない方を追加（双方向なのでどちらも可能性あり）
                    if begin_atom != atom_idx:
                        neighbor_atoms.add(begin_atom)
                    if end_atom != atom_idx:
                        neighbor_atoms.add(end_atom)
                
                if value > 0:
                    atom_contributions_plus[atom_idx] = atom_contributions_plus.get(atom_idx, 0) + value/len(neighbor_atoms)
                else:
                    atom_contributions_minus[atom_idx] = atom_contributions_minus.get(atom_idx, 0) + value/len(neighbor_atoms)
                
                # 半径 `rad` 内の隣接原子を取得
                env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    neighbor_atom_idx = bond.GetBeginAtomIdx() if bond.GetEndAtomIdx() == atom_idx else bond.GetEndAtomIdx()
                    
                    # 隣接原子にも同じ影響を与える
                    if value > 0:
                        atom_contributions_plus[neighbor_atom_idx] = atom_contributions_plus.get(neighbor_atom_idx, 0) + value
                    else:
                        atom_contributions_minus[neighbor_atom_idx] = atom_contributions_minus.get(neighbor_atom_idx, 0) + value

    # 分子を描画
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)

    # 最大 SHAP 値を取得（0 除外）
    max_shap = max(
    max(atom_contributions_plus.values(), default=0),  # 正の SHAP 値の最大
    abs(min(atom_contributions_minus.values(), default=0))  # 負の SHAP 値の最小（絶対値を取る）
)

    # 色の設定（正: 赤, 負: 青, 強さに応じて透明度変化）
    atom_colors = {}

    for idx in set(atom_contributions_plus.keys()).union(atom_contributions_minus.keys()):
        shap_plus = atom_contributions_plus.get(idx, 0)  # 正の SHAP 値
        shap_minus = atom_contributions_minus.get(idx, 0)  # 負の SHAP 値

        intensity_plus = (shap_plus / max_shap) if max_shap > 0 else 0
        intensity_minus = (abs(shap_minus) / max_shap) if max_shap > 0 else 0

        # 色相 (H) の決定
        if intensity_plus > intensity_minus:
            H = 0  # 赤
        elif intensity_minus > intensity_plus:
            H = 240  # 青
        else:
            H = 120  # 紫

        # 彩度 (S) の調整：SHAP値が0に近いほど小さく（白く）する
        S = max(intensity_plus, intensity_minus)  # SHAP値が0なら S=0（白）、最大なら S=1

        # 明度 (V) は 1 に固定（彩度だけで色の強さを調整）
        V = 1

        # HSV → RGB 変換
        r, g, b = colorsys.hsv_to_rgb(H / 360, S, V)

        # 透明度 (α) の調整
        alpha = max(S, 0.1)  # 彩度が低いほど透明度も下げる（最小0.1）

        # RGBA カラー辞書に追加
        atom_colors[idx] = (r, g, b, alpha)
    
    # 描画
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return svg