# 2022年度第4ターム 実験数学D

## これは何?

2022年度第4ターム開講「実験数学D」の講義内で紹介したJupyter NotebookおよびPythonファイルをまとめている．

## 使い方 (2023/01/08現在)

**Windowsユーザ** はPowerShellを起動して以下のコマンドを実行し，PowerShellからコマンドを実行する許可を与える:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

※自身のコンピュータ (ローカル環境) を汚したくない人はこの節をスキップし，次節の [【発展】仮想環境のすゝめ](https://github.com/fumiyanll23/2022-experimental-mathematics-c/tree/main#%E7%99%BA%E5%B1%95%E4%BB%AE%E6%83%B3%E7%92%B0%E5%A2%83%E3%81%AE%E3%81%99%E3%82%9D%E3%82%81) に従ってください．

### Anacondaを使っている場合

`conda` コマンドを利用してモジュールを管理する．

1. Anaconda Promptを起動する．

2. 以下のコマンドを実行し，必要なパッケージレポジトリを追加する:

```terminal
conda config --add channels conda-forge
```

3. 以下のコマンドを実行し，必要な外部モジュールをインストールする:

```terminal
conda install ipykernel scipy numpy nptyping
```

### それ以外の場合

`pip` コマンドを利用してモジュールを管理する．

1. Powershell (またはWindows Terminal) を起動する．

2. 以下のコマンドを実行し，必要な外部モジュールをインストールする:

```powershell
python -m pip install "ipykernel >= 6.17" "scipy >= 1.9" "numpy >= 1.23" "nptyping >= 2.4"
```

## 【発展】仮想環境のすゝめ

### Anacondaを使っている場合

`conda` コマンドを利用してConda環境を管理する．

1. Anaconda Promptを起動する．

2. 以下のコマンドを実行し，Conda環境を作成する (\``2022emd''は\``2022 Experimental Mathematics D''の略です):

```terminal
conda create --name 2022emd python=3.10
```

3. 以下のコマンドを実行し，Conda環境に入る:

```terminal
conda activate 2022emd
```

4. 以下のコマンドを実行し，必要な外部モジュールをインストールする:

```terminal
conda install ipykernel scipy numpy nptyping
```

### それ以外の場合

venvを利用して仮想環境を管理する．

1. Powershell (またはWindows Terminal) を起動する．

2. 以下のコマンドを実行し，仮想環境を作成する:

```powershell
python -m venv .venv
```

3. 以下のコマンドの **いずれか** を実行し，仮想環境に入る:

- Windowsの場合

```powershell
.venv\Scripts\activate.ps1
```

- Linux (またはWSL) の場合

```bash
source .venv/bin/activate
```

4. 以下のコマンドを実行し，必要な外部モジュールをインストールする:

```powershell
python -m pip install "ipykernel >= 6.17" "scipy >= 1.9" "numpy >= 1.23" "nptyping >= 2.4"
```

## 参考文献

- [【Anacondaの使い方】よく使うcondaコマンド一覧【チートシート】/パッケージの管理](https://hogelog.com/python/conda-command.html#toc7)
- [Python環境構築ガイド/Windows環境のPython/pip](https://www.python.jp/install/windows/pip.html)
- [Python環境構築ガイド/Anacondaのインストール/Conda コマンド](https://www.python.jp/install/anaconda/conda.html)
- [Python環境構築ガイド/Windows環境のPython/仮想環境](https://www.python.jp/install/windows/venv.html)
