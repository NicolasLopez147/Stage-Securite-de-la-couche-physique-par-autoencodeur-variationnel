import argparse
from pathlib import Path
import pandas as pd
import numpy as np

SNR_OBJETIVO = [8, 16, 20, 32]

def cargar_y_limpiar(fp: Path) -> pd.DataFrame:
    # Lee CSV, normaliza nombres de columnas y limpia filas ruidosas.
    df = pd.read_csv(fp)
    # Normaliza nombres y espacios
    df.columns = [c.strip().lower() for c in df.columns]
    # Intentamos mapear nombres comunes
    posibles = {
        "snr": [c for c in df.columns if "snr" in c],
        "epoch": [c for c in df.columns if "epoch" in c],
        "ber": [c for c in df.columns if c.strip() == "ber" or "ber" in c]
    }
    col_snr = posibles["snr"][0] if posibles["snr"] else "snr"
    col_epoch = posibles["epoch"][0] if posibles["epoch"] else "epoch"
    col_ber = posibles["ber"][0] if posibles["ber"] else "ber"

    # Si faltan, renombra o lanza error claro
    ren = {}
    if col_snr != "snr": ren[col_snr] = "snr"
    if col_epoch != "epoch": ren[col_epoch] = "epoch"
    if col_ber != "ber": ren[col_ber] = "ber"
    if ren:
        df = df.rename(columns=ren)

    # Mantener solo las columnas necesarias
    faltantes = {"snr", "epoch", "ber"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")

    df = df[["snr", "epoch", "ber"]].copy()

    # Coaccionar tipos (descarta filas basura como '........')
    df["snr"] = pd.to_numeric(df["snr"], errors="coerce")
    # Si epoch son números, los convertimos; si no, se mantienen como string
    maybe_epoch_num = pd.to_numeric(df["epoch"], errors="coerce")
    if not maybe_epoch_num.isna().all():
        df["epoch"] = maybe_epoch_num
    df["ber"] = pd.to_numeric(df["ber"], errors="coerce")

    # Filtra por SNR objetivo y elimina filas sin datos válidos
    df = df[df["snr"].isin(SNR_OBJETIVO)]
    df = df.dropna(subset=["epoch"])  # necesitamos epoch válida
    # Si hay duplicados (snr, epoch), nos quedamos con la última aparición
    df = df.sort_index().drop_duplicates(subset=["snr", "epoch"], keep="last")

    return df

def pivotear(df: pd.DataFrame) -> pd.DataFrame:
    ancho = df.pivot_table(index="epoch", columns="snr", values="ber", aggfunc="first")
    # Asegurar columnas para todos los SNR objetivo
    for s in SNR_OBJETIVO:
        if s not in ancho.columns:
            ancho[s] = np.nan
    # Ordenar columnas por SNR y renombrar
    ancho = ancho[SNR_OBJETIVO]
    ancho.columns = [f"ber_{int(c)}" for c in ancho.columns]
    # Epoch como columna normal y ordenar por epoch si es numérico
    ancho = ancho.reset_index()
    if pd.api.types.is_numeric_dtype(ancho["epoch"]):
        ancho = ancho.sort_values("epoch")
    # Reordenar columnas: epoch primero
    cols = ["epoch"] + [f"ber_{s}" for s in SNR_OBJETIVO]
    ancho = ancho[cols]
    return ancho

def validar(df: pd.DataFrame, esperado_por_snr: int = 4000):
    # Validación opcional: contar epochs por SNR
    conteos = df.groupby("snr")["epoch"].nunique()
    faltantes = []
    for s in SNR_OBJETIVO:
        real = int(conteos.get(s, 0))
        if real != esperado_por_snr:
            faltantes.append((s, real))
    if faltantes:
        avisos = ", ".join([f"SNR {s}: {c} epochs" for s, c in faltantes])
        print(f"[AVISO] Recuento de epochs distinto de {esperado_por_snr}: {avisos}")

def main():
    ap = argparse.ArgumentParser(description="Convierte CSV apilado (snr, epoch, ber) a formato ancho (epoch + ber_8, ber_16, ber_20, ber_32).")
    ap.add_argument("input_csv", type=Path, help="Ruta del CSV de entrada")
    ap.add_argument("-o", "--output", type=Path, default=Path("salida_5_columnas.csv"), help="Ruta del CSV de salida")
    ap.add_argument("--sin-validar", action="store_true", help="No mostrar validaciones de conteo por SNR")
    args = ap.parse_args()

    df = cargar_y_limpiar(args.input_csv)
    if not args.sin_validar:
        validar(df, esperado_por_snr=4000)
    ancho = pivotear(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ancho.to_csv(args.output, index=False)
    print(f"OK. Guardado en: {args.output.resolve()}")
    print("Previsualización:")
    print(ancho.head().to_string(index=False))

if __name__ == "__main__":
    main()
