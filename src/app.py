# importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import auto_arima

# cargo datos
cpu_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', index_col = 0)
cpu_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', index_col = 0)
cpu_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', index_col = 0)
cpu_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', index_col = 0)

# busco mejor combinación de parámetros
stepwise_model = auto_arima(cpu_train_a, start_p = 0, start_q = 0,
                           max_p = 6, max_q = 6, m = 4,
                           start_P = 0, seasonal = True,
                           d = 1, D = 1, trace = True,
                           error_action = 'ignore',  
                           suppress_warnings = True, 
                           stepwise = True)
print(stepwise_model.aic())

stepwise_model.fit(cpu_train_a)

# predicción de cpu_a
pred_a = stepwise_model.predict(n_periods = 60)

# ajusto el mismo modelo con otros datos (train_b)
stepwise_model.fit(cpu_train_b)

# predigo 60 obs con nuevo modelo
pred_b = stepwise_model.predict(n_periods = 60)