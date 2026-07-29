from pathlib import Path

import pandas as pd


# 以脚本所在的 Data 文件夹为基准，因此本地和服务器都能使用同一份代码。
DATA_DIR = Path(__file__).resolve().parent
INPUT_DIR = DATA_DIR / "raw" / "cgm"
OUTPUT_DIR = DATA_DIR / "csv" / "cgm"


def convert_all_parquet() -> None:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(
        file
        for file in INPUT_DIR.iterdir()
        if file.is_file() and file.suffix.lower() == ".parquet"
    )

    if not parquet_files:
        print(f"输入文件夹中没有 Parquet 文件：{INPUT_DIR}")
        return

    print(f"共找到 {len(parquet_files)} 个 Parquet 文件")

    for parquet_path in parquet_files:
        csv_path = OUTPUT_DIR / f"{parquet_path.stem}.csv"

        print(f"正在转换：{parquet_path.name}")
        dataframe = pd.read_parquet(parquet_path)

        dataframe.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"转换完成：{csv_path.name}，"
            f"{len(dataframe):,} 行，"
            f"{len(dataframe.columns)} 列"
        )

        del dataframe

    print(f"全部转换完成，输出文件夹：{OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all_parquet()
