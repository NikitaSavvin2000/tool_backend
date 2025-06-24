import os
import pandas as pd
from src.xgboost_selection_of_parameters.main import user_predict_XGBoost
from src.utils.date_utils import standardize_datetime

if __name__ == "__main__":
    # Тестовый случай с 2 строками
    test_df = pd.DataFrame({
        "Дата": ["2023-01-01 00:00:00", "2023-01-02 00:00:00"],
        "Цена": [100.0, 105.0],
    })

    # Преобразование даты
    test_df["Дата"] = test_df["Дата"].apply(lambda x: standardize_datetime(str(x)))

    # Параметры прогнозирования
    time_column = "Дата"
    col_target = "Цена"
    forecast_horizon_time = "2023-01-03 00:00:00"
    forecast_horizon_time = standardize_datetime(forecast_horizon_time)

    # Прогнозирование
    try:
        predict_dict = user_predict_XGBoost(
            df=test_df,
            time_column=time_column,
            col_target=col_target,
            forecast_horizon_time=forecast_horizon_time,
        )
        print("Прогноз успешно построен:")
        print(predict_dict)
    except ValueError as e:
        print(f"Ошибка: {e}")

    # Основной запуск с реальными данными
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Прошлые данные - TSLA.csv")
    
    # Загружаем данные
    df_data = pd.read_csv(csv_path)
    
    # Очистка данных
    cols_to_convert = ['Откр.', 'Макс.', 'Мин.', 'Цена', 'Объём', 'Изм. %']
    df_data[cols_to_convert] = df_data[cols_to_convert].apply(
        lambda x: x.map(lambda y: float(str(y).replace('%', '').replace('M', '').replace(',', '.')))
    )
    
    # Преобразование даты
    df_data['Дата'] = df_data['Дата'].apply(lambda x: standardize_datetime(str(x)))
    
    # Параметры прогнозирования
    time_column = 'Дата'
    col_target = 'Цена'
    forecast_horizon_time = '2025-06-05 00:00:00'
    forecast_horizon_time = standardize_datetime(forecast_horizon_time)
    
    # Разделение данных на обучающую и тестовую выборки
    df_to_predict = df_data.iloc[:-30]  # Обучающая выборка
    df_test = df_data.iloc[-30:]       # Тестовая выборка
    
    # Прогнозирование
    predict_dict = user_predict_XGBoost(
        df=df_to_predict,
        time_column=time_column,
        col_target=col_target,
        forecast_horizon_time=forecast_horizon_time,
    )
    
    # Извлечение прогнозов
    predictions = predict_dict["map_data"]["data"]["predictions"]
    df_predictions = pd.DataFrame(predictions).iloc[1:]  # Удаление первой строки (последнее известное значение)
    df_predictions = df_predictions.head(len(df_test))  # Совмещение с тестовой выборкой
    
    # Стандартизация временных меток в прогнозах
    df_predictions[time_column] = df_predictions[time_column].apply(lambda x: standardize_datetime(str(x)))
    
    # Вывод результатов
    print("Прогнозные значения:")
    print(df_predictions)
    
    # Расчет метрик
    y_true = df_test[col_target].values
    y_pred = df_predictions[col_target].values  # Предполагается, что второй столбец содержит прогнозы
    
    from src.utils.metrics import calculate_metrics
    rmse, r2, mae, mape, wmape = calculate_metrics(y_true=y_true, y_pred=y_pred)
    print(f"RMSE: {rmse}, R2: {r2}, MAE: {mae}, MAPE: {mape}, WMAPE: {wmape}")