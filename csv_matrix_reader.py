#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルから行列の値をnumpy.arrayに読み込むプログラム
行と列の開始位置と行列の大きさをマクロで指定可能
ベクトルとの内積演算と3Dグラフィカル表示機能付き
"""

import numpy as np
import pandas as pd
import sys
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib import font_manager

# グローバル変数で日本語フォント名を保持
JP_FONT_NAME = None

# 日本語フォントの設定
def setup_japanese_font():
    global JP_FONT_NAME
    """日本語フォントを設定する"""
    # 利用可能な日本語フォントを探す
    japanese_fonts = [
        'Noto Sans CJK JP', 'IPAGothic', 'VL Gothic', 'TakaoGothic',
        'Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'Source Han Sans JP',
        'Arial Unicode MS', 'DejaVu Sans'  # DejaVuは最後
    ]
    
    # システムにインストールされているフォントを確認
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 日本語フォントを設定
    for font in japanese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            JP_FONT_NAME = font
            print(f"日本語フォント '{font}' を使用します")
            return True
    
    # フォールバック: デフォルトフォントを使用
    print("警告: 日本語フォントが見つかりません。デフォルトフォントを使用します")
    JP_FONT_NAME = None
    return False

# プログラム開始時に日本語フォントを設定
setup_japanese_font()

# マクロ定義
# 行列の読み込み設定
MATRIX_START_ROW = 1      # 行列の開始行（0から開始）
MATRIX_START_COL = 1      # 行列の開始列（0から開始）
MATRIX_ROWS = 3           # 行列の行数
MATRIX_COLS = 3           # 行列の列数

# ベクトルの読み込み設定
VECTOR_START_ROW = 1      # ベクトルの開始行（0から開始）
VECTOR_START_COL = 5      # ベクトルの開始列（0から開始）- 右端の列
VECTOR_LENGTH = 3         # ベクトルの長さ

def read_matrix_from_csv(
    csv_file_path: str,
    start_row: int = MATRIX_START_ROW,
    start_col: int = MATRIX_START_COL,
    rows: int = MATRIX_ROWS,
    cols: int = MATRIX_COLS,
    delimiter: str = ',',
    header: Optional[int] = None,
    index_col: Optional[int] = None
) -> np.ndarray:
    """
    CSVファイルから行列の値をnumpy.arrayに読み込む
    
    Parameters:
    -----------
    csv_file_path : str
        CSVファイルのパス
    start_row : int
        読み込み開始行（0から開始）
    start_col : int
        読み込み開始列（0から開始）
    rows : int
        読み込む行数
    cols : int
        読み込む列数
    delimiter : str
        CSVの区切り文字（デフォルト: ','）
    header : int, optional
        ヘッダー行の位置（Noneの場合はヘッダーなし）
    index_col : int, optional
        インデックス列の位置（Noneの場合はインデックスなし）
    
    Returns:
    --------
    np.ndarray
        読み込んだ行列データ
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(
            csv_file_path,
            delimiter=delimiter,
            header=header,
            index_col=index_col
        )
        
        # 終了位置を計算
        end_row = start_row + rows
        end_col = start_col + cols
        
        # 範囲チェック
        if start_row < 0 or start_col < 0:
            raise ValueError("開始行・列は0以上である必要があります")
        if start_row >= len(df) or start_col >= len(df.columns):
            raise ValueError("開始行・列がデータの範囲外です")
        if end_row > len(df) or end_col > len(df.columns):
            raise ValueError("指定された行列サイズがデータの範囲外です")
        if rows <= 0 or cols <= 0:
            raise ValueError("行列の行数・列数は1以上である必要があります")
        
        # 指定された範囲のデータを抽出
        selected_data = df.iloc[start_row:end_row, start_col:end_col]
        
        # numpy.arrayに変換
        matrix = selected_data.to_numpy()
        matrix = matrix.astype(float)
        
        return matrix
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_file_path}' が見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

def read_vector_from_csv(
    csv_file_path: str,
    start_row: int = VECTOR_START_ROW,
    start_col: int = VECTOR_START_COL,
    length: int = VECTOR_LENGTH,
    delimiter: str = ',',
    header: Optional[int] = None,
    index_col: Optional[int] = None
) -> np.ndarray:
    """
    CSVファイルから列ベクトルの値をnumpy.arrayに読み込む
    
    Parameters:
    -----------
    csv_file_path : str
        CSVファイルのパス
    start_row : int
        読み込み開始行（0から開始）
    start_col : int
        読み込み開始列（0から開始）
    length : int
        ベクトルの長さ
    delimiter : str
        CSVの区切り文字（デフォルト: ','）
    header : int, optional
        ヘッダー行の位置（Noneの場合はヘッダーなし）
    index_col : int, optional
        インデックス列の位置（Noneの場合はインデックスなし）
    
    Returns:
    --------
    np.ndarray
        読み込んだベクトルデータ（1次元配列）
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(
            csv_file_path,
            delimiter=delimiter,
            header=header,
            index_col=index_col
        )
        
        # 終了位置を計算（列ベクトルなので行方向にlength分）
        end_row = start_row + length
        end_col = start_col + 1
        
        # 範囲チェック
        if start_row < 0 or start_col < 0:
            raise ValueError("開始行・列は0以上である必要があります")
        if start_row >= len(df) or start_col >= len(df.columns):
            raise ValueError("開始行・列がデータの範囲外です")
        if end_row > len(df) or end_col > len(df.columns):
            raise ValueError("指定されたベクトルサイズがデータの範囲外です")
        if length <= 0:
            raise ValueError("ベクトルの長さは1以上である必要があります")
        
        # 指定された範囲のデータを抽出（列ベクトルとして）
        selected_data = df.iloc[start_row:end_row, start_col:end_col]
        
        # numpy.arrayに変換して1次元配列に
        vector = selected_data.to_numpy().flatten()
        vector = vector.astype(float)
        
        return vector
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_file_path}' が見つかりません")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

def matrix_vector_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    行列とベクトルの内積演算
    
    Parameters:
    -----------
    matrix : np.ndarray
        行列
    vector : np.ndarray
        ベクトル
    
    Returns:
    --------
    np.ndarray
        内積演算の結果
    """
    try:
       result = np.dot(matrix, vector)
       return result
    
    except ValueError as e:
        print(f"行列とベクトルのサイズが合いません: {e}")
        sys.exit(1)

def calculate_vector_info(vector: np.ndarray, vector_name: str = "ベクトル") -> dict:
    """
    ベクトルの情報を計算する
    
    Parameters:
    -----------
    vector : np.ndarray
        ベクトル
    vector_name : str
        ベクトルの名前
    
    Returns:
    --------
    dict
        ベクトルの情報（長さ、角度など）
    """
    # ベクトルの長さ（ノルム）
    length = np.linalg.norm(vector)
    
    # 各軸との角度（ラジアン）
    if length > 0:
        angles = {
            'x_angle': np.arccos(np.abs(vector[0]) / length),
            'y_angle': np.arccos(np.abs(vector[1]) / length),
            'z_angle': np.arccos(np.abs(vector[2]) / length)
        }
    else:
        angles = {'x_angle': 0, 'y_angle': 0, 'z_angle': 0}
    
    return {
        'name': vector_name,
        'vector': vector,
        'length': length,
        'angles': angles
    }

def plot_3d_vectors(vector: np.ndarray, title: str, x:str, y:str, z:str):
    """
    3D座標でベクトルをグラフィカル表示（原点中心）
    
    Parameters:
    -----------
    vector : np.ndarray
        表示するベクトル
    matrix : np.ndarray
        使用した行列
    title : str
        グラフのタイトル
    """
    setup_japanese_font()
    font_prop = None
    if JP_FONT_NAME:
        font_prop = font_manager.FontProperties(family=JP_FONT_NAME)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原点
    origin = np.array([0, 0, 0])
    
    # 座標軸の最大値を計算
    max_val = np.max(np.abs(vector))
    
    # 座標軸の範囲を設定（原点中心）
    ax.set_xlim([-max_val*0, max_val*1.2])
    ax.set_ylim([-max_val*0, max_val*1.2])
    ax.set_zlim([-max_val*0, max_val*1.2])
    
    # 座標軸を表示
    ax.set_xlabel('X:'+x, fontsize=12, fontproperties=font_prop)
    ax.set_ylabel('Y:'+y, fontsize=12, fontproperties=font_prop)
    ax.set_zlabel('Z:'+z, fontsize=12, fontproperties=font_prop)
    
    # 原点に点を表示
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin')
    
    # ベクトル情報を計算
    vector_info = calculate_vector_info(vector, "Vector")
   
    # 元のベクトル（青色）- 原点からスタート
    ax.quiver(origin[0], origin[1], origin[2], 
              vector[0], vector[1], vector[2],
              color='blue', arrow_length_ratio=0.15, linewidth=3, 
              alpha=0.8, label=f'Vector: {vector}')
    
    
    # 座標軸の線を表示
    ax.plot([0, max_val*1.1], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, max_val*1.1], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], [0, max_val*1.1], 'k-', linewidth=1, alpha=0.3)
    
    # 負の座標軸も表示
    ax.plot([0, -max_val*1.1], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, -max_val*1.1], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], [0, -max_val*1.1], 'k-', linewidth=1, alpha=0.3)
    
    # タイトルと凡例
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=font_prop)
    ax.legend(fontsize=10, loc='upper right', prop=font_prop)
    
    # グリッドを表示
    ax.grid(True, alpha=0.3)
    
    # アスペクト比を調整
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    # ベクトル情報をテキストで表示
    info_text = f"""
ベクトル情報:
ベクトル: {vector}
  長さ: {vector_info['length']:.3f}
  角度: X軸({np.degrees(vector_info['angles']['x_angle']):.1f}°), Y軸({np.degrees(vector_info['angles']['y_angle']):.1f}°), Z軸({np.degrees(vector_info['angles']['z_angle']):.1f}°)
"""
    print(info_text)
    
    plt.show()

def print_matrix_info(matrix: np.ndarray, matrix_name: str = "行列"):
    """
    行列の情報を表示する
    
    Parameters:
    -----------
    matrix : np.ndarray
        表示する行列
    matrix_name : str
        行列の名前
    """
    print(f"\n{matrix_name}の情報:")
    print(f"  形状: {matrix.shape}")
    print(f"  データ型: {matrix.dtype}")
    print(f"  要素数: {matrix.size}")
    print(f"  内容:")
    print(matrix)
    print()

def print_vector_info(vector: np.ndarray, vector_name: str = "ベクトル"):
    """
    ベクトルの情報を表示する
    
    Parameters:
    -----------
    vector : np.ndarray
        表示するベクトル
    vector_name : str
        ベクトルの名前
    """
    print(f"\n{vector_name}の情報:")
    print(f"  長さ: {len(vector)}")
    print(f"  データ型: {vector.dtype}")
    print(f"  内容: {vector}")
    print()

def main():
    """
    メイン関数
    """
    # 使用例
    csv_file = "sample_matrix.csv"
    
    print("CSVファイルから行列とベクトルを読み込みます...")
    print(f"ファイル: {csv_file}")
    print(f"行列設定: 開始位置({MATRIX_START_ROW}, {MATRIX_START_COL}), サイズ({MATRIX_ROWS}x{MATRIX_COLS})")
    print(f"ベクトル設定: 開始位置({VECTOR_START_ROW}, {VECTOR_START_COL}), 長さ({VECTOR_LENGTH})")
    
    # 行列を読み込み
    matrix = read_matrix_from_csv(csv_file)
    print_matrix_info(matrix, "読み込んだ行列")
    
    # ベクトルを読み込み
    vector = read_vector_from_csv(csv_file)
    print_vector_info(vector, "読み込んだベクトル")
    
    # 行列とベクトルの内積演算
    print("行列とベクトルの内積演算を実行...")
    result = matrix_vector_product(matrix, vector)
    print_vector_info(result, "演算結果")
    
    # 3Dグラフィカル表示
    print("3Dグラフィカル表示を生成中...")
    plot_3d_vectors(vector, "Input", "Mirin", "Sauce", "Shouyu")
    plot_3d_vectors(result, "Result","Sweet", "Salty", "Sour")

if __name__ == "__main__":
    main() 