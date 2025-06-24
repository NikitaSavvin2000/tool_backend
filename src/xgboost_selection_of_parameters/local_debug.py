# local_debug.py

import os
import pandas as pd
from src.xgboost_selection_of_parameters.main import user_predict_XGBoost

if __name__ == "__main__":
    # Получаем путь к директории, где находится local_debug.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к CSV файлу с данными
    csv_path = os.path.join(current_dir, "Прошлые данные - TSLA.csv")
    
    # Загружаем данные
    df_data = pd.read_csv(csv_path)
    
    # Очистка данных
    cols_to_convert = ['Откр.', 'Макс.', 'Мин.', 'Цена', 'Объём', 'Изм. %']
    df_data[cols_to_convert] = df_data[cols_to_convert].applymap(
        lambda x: float(str(x).replace('%', '').replace('M', '').replace(',', '.'))
    )
    
    # Преобразование даты
    df_data['Дата'] = pd.to_datetime(df_data['Дата'], format='%d.%m.%Y')
    df_data['Дата'] = df_data['Дата'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Параметры прогнозирования
    time_column = 'Дата'
    col_target = 'Цена'
    forecast_horizon_time = '2025-06-05 00:00:00'
    
    # Разделение данных на обучающую и тестовую выборки
    df_to_predict = df_data.iloc[:-30]  # Обучающая выборка
    df_test = df_data.iloc[-30:]       # Тестовая выборка
    
    # Прогнозирование
    predict_dict = user_predict_XGBoost(
        df=df_to_predict,
        time_column=time_column,
        col_target=col_target,
        forecast_horizon_time=forecast_horizon_time
    )
    
    # Извлечение прогнозов
    predictions = predict_dict["map_data"]["data"]["predictions"]
    df_predictions = pd.DataFrame(predictions).iloc[1:]  # skip last known value
    df_predictions = df_predictions.head(len(df_test))  # align with df_test
    
    # Вывод результатов
    print("Прогнозные значения:")
    print(df_predictions)
    
    # Расчет метрик (если необходимо)
    y_true = df_test[col_target].values
    y_pred = df_predictions.iloc[:, 1].values  
    
    from src.utils.metrics import calculate_metrics
    rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)
    print(f"RMSE: {rmse}, R2: {r2}, MAE: {mae}, MAPE: {mape}, WMAPE: {wmape}")