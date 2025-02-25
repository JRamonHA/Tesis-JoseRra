import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

def scatter_plot(filepath, columns):
    """
    Genera un scatter plot a partir de un archivo Parquet usando las columnas especificadas.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - columns (list): Lista de nombres de columnas a graficar en el eje Y.

    Retorna:
    - None: Muestra el scatter plot interactivo.
    """
    data = pd.read_parquet(filepath)
    fig = px.scatter(data, x=data.index, y=columns)
    fig.update_layout(hovermode="x unified")
    fig.show()


def calculate_error(filepath, col1, col2, resample_interval=None):
    """
    Calcula el Error Medio (ME) y el Error Absoluto Medio (MAE) entre dos columnas.
    Opcionalmente, permite resamplear el DataFrame antes de calcular los errores.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - col1 (str): Nombre de la primera columna.
    - col2 (str): Nombre de la segunda columna.
    - resample_interval (str, opcional): Intervalo de resampleo (ej. '10s'). Si no se proporciona, no se resamplea.

    Retorna:
    - tuple: (ME, MAE)
    """
    data = pd.read_parquet(filepath)

    if col1 not in data.columns or col2 not in data.columns:
        raise ValueError(f"Las columnas {col1} y {col2} no existen en el DataFrame.")

    if resample_interval:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex para resamplear.")
        data = data.resample(resample_interval).mean()

    error = data[col1] - data[col2]
    ME = error.mean()
    MAE = error.abs().mean()
    return ME, MAE


def wind_calibration(filepath, ws='WS', v='V', t='T', resample_interval=None):
    """
    Ajusta la ecuación de calibración del Wind Sensor.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - ws (str): Nombre de la columna con la velocidad del viento (WS).
    - v (str): Nombre de la columna con el voltaje ajustado (V).
    - t (str): Nombre de la columna con la temperatura (T).
    - resample_interval (str, opcional): Intervalo de resampleo (ej. '10s'). Si no se proporciona, no se resamplea.

    Retorna:
    - dict: Diccionario con los parámetros de calibración.
    """
    data = pd.read_parquet(filepath)

    if not all(col in data.columns for col in [ws, v, t]):
        raise ValueError("Una o más columnas especificadas no existen en el DataFrame.")

    if resample_interval:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex para resamplear.")
        data = data.resample(resample_interval).mean()

    data_filtered = data[(data[ws] > 0) & (data[v] > 0) & (data[t] > 0)]
    X = np.log(data_filtered[[v, t]])
    y = np.log(data_filtered[ws])
    modelo = LinearRegression()
    modelo.fit(X, y)
    intercepto = modelo.intercept_
    coeficientes = modelo.coef_
    a = np.exp(intercepto)
    b = coeficientes[0]
    c = coeficientes[1]
    equation = f"WS = {a:.4f} * {v}^({b:.4f}) * {t}^({c:.4f})"
    r2 = modelo.score(X, y)
    return {"a": a, "b": b, "c": c, "r2": r2, "equation": equation}


def thermo_calibration(filepath, columns, time_intervals):
    """
    Realiza ecuaciones de calibración para termopares a partir 
    de columnas específicas de un DataFrame e intervalos de tiempo, 
    asignando valores a 'Tref' y realizando regresiones.

    Parámetros:
    - filepath (str): Ruta al archivo Parquet.
    - columns (list): Lista de nombres de columnas para la regresión.
    - time_intervals (list): Lista de tuplas (inicio, fin, valor_Tref).

    Retorna:
    - None: Imprime las ecuaciones de calibración para cada columna.
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression

    df = pd.read_parquet(filepath)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    df["Tref"] = np.nan
    for start, end, tref_value in time_intervals:
        indices = df.between_time(start, end).index
        df.loc[indices, "Tref"] = tref_value

    data_all = df.dropna(subset=["Tref"])
    for col in columns:
        temp = data_all[[col, 'Tref']].dropna()
        X = temp[col].values.reshape(-1, 1)
        y = temp['Tref'].values
        modelo = LinearRegression()
        modelo.fit(X, y)
        coef = modelo.coef_[0]
        intercepto = modelo.intercept_
        print(f'Para la columna {col}:')
        print(f'  Ecuación de calibración: Tref = {coef:.4f} * {col} + {intercepto:.4f}\n')