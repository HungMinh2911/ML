import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.express as px  # vẽ biểu đồ tương tác

housing_data_dataset = pd.read_csv(r"C:\Users\PC\Downloads\HousingData.csv")

training_data = housing_data_dataset.loc[:, ('MEDV','RM','LSTAT','NOX','DIS','CRIM')]

# Kiểm tra NA
print("Số lượng NA theo cột:\n", training_data.isna().sum())

# Xử lý NA trong cột LSTAT (dùng mean)
dataset = training_data.copy()
dataset.loc[:, "LSTAT"] = dataset["LSTAT"].fillna(dataset["LSTAT"].mean())

# Tính hệ số tương quan
print(dataset.corr(numeric_only=True))

# Vẽ scatter matrix
fig = px.scatter_matrix(dataset, dimensions=['MEDV','RM','LSTAT','NOX','DIS','CRIM'])
# fig.show()

# ------------------- Định nghĩa class và hàm -------------------
class ExperimentSettings:
    def __init__(self, learning_rate=0.001, number_epochs=20, batch_size=50, input_features=None):
        self.learning_rate = learning_rate
        self.number_epochs = number_epochs
        self.batch_size = batch_size
        self.input_features = input_features if input_features is not None else []

def create_model(settings: ExperimentSettings, metrics: list) -> keras.Model:
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    noi_inputs = keras.layers.Concatenate()(list(inputs.values()))
    output = keras.layers.Dense(units=1)(noi_inputs)
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(settings.learning_rate),
        loss='mean_squared_error',
        metrics=metrics
    )
    return model

class Experiment:
    def __init__(self, name, settings, model, epochs, metric_history):
        self.name = name
        self.settings = settings
        self.model = model
        self.epochs = epochs
        self.metric_history = metric_history

def train_model(experiment_name: str, model: keras.Model, dataset: pd.DataFrame, label_name: str, settings: ExperimentSettings) -> Experiment:
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values

    history = model.fit(
        x=features,
        y=label,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )
    return Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metric_history=pd.DataFrame(history.history),
    )

# ------------------- Chạy thử -------------------
settings_1 = ExperimentSettings(
    learning_rate=0.1,
    number_epochs=200,
    batch_size=100,
    input_features=['LSTAT']
)
metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
model = create_model(settings_1, metrics)
experiment_1 = train_model('So sanh ', model, dataset, 'MEDV', settings_1)

def plot_experiment_metrics(experiment, metrics):
    """Vẽ biểu đồ metrics đơn giản"""
    plt.figure(figsize=(8, 5))
    for metric in metrics: # Lặp qua danh sách các metrics cần vẽ
        if metric in experiment.metric_history.columns:             # Nếu metric có trong dữ liệu lịch sử (giống DataFrame với các cột)
            # Nếu metric có trong dữ liệu lịch sử (giống DataFrame với các cột)

            plt.plot(experiment.epochs, # truc X : số epoch
                     experiment.metric_history[metric], # trục Y : giá trị metric theo từng epoch
                     label=metric.upper())
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'{experiment.name} - Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()
class SimpleMLEdu:
    class results:
        plot_experiment_metrics = staticmethod(plot_experiment_metrics)

ml_edu = SimpleMLEdu()

ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])