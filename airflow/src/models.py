# ===================================================================
# MÓDULO: models.py
# ===================================================================

# Importações de bibliotecas de terceiros
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from darts import TimeSeries
from darts.models import NBEATSModel, RNNModel
from darts.dataprocessing.transformers import Scaler

# Importações dos nossos módulos locais
from src.framework import BaseModel, get_safe_pandas_series

# --- Definição dos Modelos Puros (Baselines) ---

class PureARIMA(BaseModel):
    """
    Um wrapper para o modelo AutoARIMA da biblioteca pmdarima.
    Este modelo serve como um baseline estatístico robusto.
    """
    def __init__(self, name="PureARIMA"):
        super().__init__(name)
        self.model = None

    def fit(self, train_series: TimeSeries, forecast_horizon: int):
        """
        Ajusta o modelo AutoARIMA aos dados de treinamento.
        A sazonalidade é detectada automaticamente com m=12.
        """
        self.model = auto_arima(
            get_safe_pandas_series(train_series),
            seasonal=True, m=12,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )

    def predict(self, n: int) -> TimeSeries:
        """
        Realiza a previsão para os próximos 'n' passos.
        """
        return TimeSeries.from_series(self.model.predict(n_periods=n))


class PureLSTM(BaseModel):
    def __init__(self, name="PureLSTM", n_lags=24, n_epochs=100, **kwargs): # Adicionado **kwargs
        super().__init__(name)
        self.n_lags = n_lags
        self.n_epochs = n_epochs
        self.scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        self.model = None
        self.kwargs = kwargs # Armazena os parâmetros extras

    def fit(self, train_series: TimeSeries, forecast_horizon: int):
        train_scaled = self.scaler.fit_transform(train_series)
        self.model = RNNModel(
            model='LSTM',
            input_chunk_length=self.n_lags,
            output_chunk_length=1,
            n_epochs=self.n_epochs,
            random_state=42,
            n_rnn_layers=2,
            **self.kwargs # Repassa os parâmetros extras como batch_size
        )
        self.model.fit(train_scaled)

    def predict(self, n: int) -> TimeSeries:
        prediction_scaled = self.model.predict(n=n)
        return self.scaler.inverse_transform(prediction_scaled)


class PureNBEATS(BaseModel):
    def __init__(self, name="PureNBEATS", n_lags=24, n_epochs=100, **kwargs): # Adicionado **kwargs
        super().__init__(name)
        self.n_lags = n_lags
        self.n_epochs = n_epochs
        self.scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        self.model = None
        self.kwargs = kwargs # Armazena os parâmetros extras

    def fit(self, train_series: TimeSeries, forecast_horizon: int):
        train_scaled = self.scaler.fit_transform(train_series)
        self.model = NBEATSModel(
            input_chunk_length=self.n_lags,
            output_chunk_length=forecast_horizon,
            n_epochs=self.n_epochs,
            random_state=42,
            **self.kwargs # Repassa os parâmetros extras como batch_size
        )
        self.model.fit(train_scaled)

    def predict(self, n: int) -> TimeSeries:
        prediction_scaled = self.model.predict(n=n)
        return self.scaler.inverse_transform(prediction_scaled)