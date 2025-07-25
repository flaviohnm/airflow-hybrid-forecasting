# ===================================================================
# MÓDULO: framework.py
# ===================================================================

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def create_sliding_window_dataset(data: np.ndarray, n_in: int = 1, n_out: int = 1) -> (np.ndarray, np.ndarray):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_in
        out_end_ix = end_ix + n_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_safe_pandas_series(darts_series: TimeSeries) -> pd.Series:
    return pd.Series(darts_series.values().flatten(), index=darts_series.time_index)

class BaseModel(ABC):
    def __init__(self, name: str, **kwargs): # Adicionado **kwargs para flexibilidade
        self.name = name

    @abstractmethod
    def fit(self, train_series: TimeSeries, forecast_horizon: int):
        pass

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        pass

    def __str__(self):
        return self.name

class MLPRegressorWrapper:
    def __init__(self, n_lags=24, output_chunk_length=12, max_iter=500, **kwargs):
        self.n_lags = n_lags
        self.output_chunk_length = output_chunk_length
        self.max_iter = max_iter
        self.model = MLPRegressor(
            hidden_layer_sizes=(20,), max_iter=self.max_iter, random_state=42, **kwargs
        )

    def fit(self, series: TimeSeries):
        X, y = create_sliding_window_dataset(
            series.values().flatten(), n_in=self.n_lags, n_out=self.output_chunk_length
        )
        self.model.fit(X, y)

    def predict(self, n: int, series: TimeSeries) -> TimeSeries:
        input_data = series[-self.n_lags:].values().flatten().reshape(1, -1)
        prediction_values = self.model.predict(input_data).flatten()
        if len(prediction_values) > n:
            prediction_values = prediction_values[:n]
        elif len(prediction_values) < n:
            prediction_values = np.pad(prediction_values, (0, n - len(prediction_values)), 'edge')
            
        start_date = series.end_time() + series.freq
        pred_index = pd.date_range(start=start_date, periods=n, freq=series.freq)
        return TimeSeries.from_times_and_values(pred_index, prediction_values)

class HybridForecastingFramework(BaseModel):
    def __init__(self, name: str, non_linear_model_class, non_linear_model_params: dict, strategy: str = 'mimo', **kwargs):
        super().__init__(name=name, **kwargs)
        self.non_linear_model_class = non_linear_model_class
        self.strategy = strategy
        self.non_linear_params = non_linear_model_params
        
        self.arima_model = None
        self.non_linear_model = None
        self.residuals_train = None
        self.scaler = Scaler(MinMaxScaler(feature_range=(-1, 1)))

    def fit(self, train_series: TimeSeries, forecast_horizon: int):
        from darts.models import NBEATSModel, RNNModel # Importação local para o worker
        
        self.arima_model = auto_arima(
            get_safe_pandas_series(train_series), seasonal=False, stepwise=True, 
            suppress_warnings=True, error_action='ignore'
        )
        residuals_pd = self.arima_model.resid()
        self.residuals_train = TimeSeries.from_series(residuals_pd)
        residuals_scaled = self.scaler.fit_transform(self.residuals_train)

        if self.strategy == 'direct':
            self.non_linear_model = {}
            for h in range(1, forecast_horizon + 1):
                expert_params = self.non_linear_params.copy()
                expert_params['output_chunk_length'] = h
                expert = self.non_linear_model_class(**expert_params)
                expert.fit(residuals_scaled)
                self.non_linear_model[h] = expert
        
        elif self.strategy in ['mimo', 'recursive']:
            expert_params = self.non_linear_params.copy()
            expert_params['output_chunk_length'] = forecast_horizon if self.strategy == 'mimo' else 1
            self.non_linear_model = self.non_linear_model_class(**expert_params)
            self.non_linear_model.fit(residuals_scaled)
            
    def predict(self, n: int) -> TimeSeries:
        arima_forecast = TimeSeries.from_series(self.arima_model.predict(n_periods=n))
        residuals_scaled = self.scaler.transform(self.residuals_train)

        if self.strategy == 'direct':
            forecasts_np = np.zeros(n)
            for h in range(1, n + 1):
                pred_h = self.non_linear_model[h].predict(n=h, series=residuals_scaled)
                forecasts_np[h-1] = pred_h.values().flatten()[-1]
            pred_ts_scaled = TimeSeries.from_times_and_values(arima_forecast.time_index, forecasts_np)
        else:
            pred_ts_scaled = self.non_linear_model.predict(n=n, series=residuals_scaled)

        residual_forecast = self.scaler.inverse_transform(pred_ts_scaled)
        
        return arima_forecast + residual_forecast