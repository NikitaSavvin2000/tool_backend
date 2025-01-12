example_dfs_1 = {
    "09:00:00": 10,
    "09:01:00": 12,
    "09:02:00": 15,
    "09:03:00": 14,
    "09:04:00": 13
}

example_dfs_2 = {
    "10:00:00": 20,
    "10:01:00": 22,
    "10:02:00": 21,
    "10:03:00": 25,
    "10:04:00": 24
}

example_dfs_3 = {
    "11:00:00": 30,
    "11:01:00": 29,
    "11:02:00": 31,
    "11:03:00": 33,
    "11:04:00": 32
}

example_not_norm_data = {
    "col_time": "time",
    "col_target": "load_consumption",
    "json_list_df": [
        {"load_consumption": 56797.7, "time": "2023-08-19 17:53:00"},
        {"load_consumption": 58040.5, "time": "2023-08-19 17:58:00"}]
}

example_reverse_norm_data = {
    "col_time": "time",
    "col_target": "load_consumption",
    "json_list_norm_df": [
        {"load_consumption": 0.8535390496253967, "week": 0.9215686274509803, "day_of_week": 0.6666666666666666,
         "hour": 0.782608695652174, "minute": 0.6610169491525424, "hour_sin": -1, "hour_cos": -1.8369701987210297e-16,
         "day_of_week_sin": -0.433883739117558, "day_of_week_cos": -0.9009688679024191, "week_sin": -0.4647231720437684,
         "week_cos": 0.88545602565321, "is_holiday": 0, "second": 0, "year": 0.968503937007874},
        {"load_consumption": 0.9265455603599548, "week": 0.9215686274509803, "day_of_week": 0.6666666666666666,
         "hour": 0.782608695652174, "minute": 0.7457627118644068, "hour_sin": -1, "hour_cos": -1.8369701987210297e-16,
         "day_of_week_sin": -0.433883739117558, "day_of_week_cos": -0.9009688679024191, "week_sin": -0.4647231720437684,
         "week_cos": 0.88545602565321, "is_holiday": 0, "second": 0, "year": 0.968503937007874},
        {"load_consumption": 1.0235118865966797, "week": 0.9215686274509803, "day_of_week": 0.6666666666666666,
         "hour": 0.782608695652174, "minute": 0.8305084745762712, "hour_sin": -1, "hour_cos": -1.8369701987210297e-16,
         "day_of_week_sin": -0.433883739117558, "day_of_week_cos": -0.9009688679024191, "week_sin": -0.4647231720437684,
         "week_cos": 0.88545602565321, "is_holiday": 0, "second": 0, "year": 0.968503937007874},
        {"load_consumption": 1.1619222164154053, "week": 0.9215686274509803, "day_of_week": 0.6666666666666666,
         "hour": 0.782608695652174, "minute": 0.9152542372881356, "hour_sin": -1, "hour_cos": -1.8369701987210297e-16,
         "day_of_week_sin": -0.433883739117558, "day_of_week_cos": -0.9009688679024191, "week_sin": -0.4647231720437684,
         "week_cos": 0.88545602565321, "is_holiday": 0, "second": 0, "year": 0.968503937007874},
        {"load_consumption": 1.3761029243469238, "week": 0.9215686274509803, "day_of_week": 0.6666666666666666,
         "hour": 0.782608695652174, "minute": 1, "hour_sin": -1, "hour_cos": -1.8369701987210297e-16,
         "day_of_week_sin": -0.433883739117558, "day_of_week_cos": -0.9009688679024191, "week_sin": -0.4647231720437684,
         "week_cos": 0.88545602565321, "is_holiday": 0, "second": 0, "year": 0.968503937007874}
    ],
    "min_val": 0.1,
    "max_val": 200000.5,
}

example_forecast_point = {
    "col_target": "load_consumption",
    "evaluation_index": 7,
    "last_know_index": 9,
    "epochs": 5,
    "lag": 1,
    "activation": "relu",
    "optimizer": "adam",
    "dropout_count": 0.01,
    "model_architecture_params": [
        {"layer": 1, "type": "Bi-LSTM", "neurons": 2},
        {"layer": 2, "type": "Bi-LSTM", "neurons": 4},
        {"layer": 3, "type": "Bi-LSTM", "neurons": 8}],

    "json_list_df_all_data_norm": [
        {"load_consumption": 0.6800409376, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7391304348, "minute": 0.8983050847, "second": 0, "hour_sin": -0.9659258263,
         "hour_cos": -0.2588190451, "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934,
         "week_sin": -0.7485107482, "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6952882419, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7391304348, "minute": 0.9830508475, "second": 0, "hour_sin": -0.9659258263,
         "hour_cos": -0.2588190451, "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934,
         "week_sin": -0.7485107482, "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.7110373283, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.0508474576, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6988007163, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.1355932203, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6925315077, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.2203389831, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6344474735, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.3050847458, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5210668107, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.3898305085, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5220188471, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.4745762712, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5324090483, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.5593220339, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5249571553, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.6440677966, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0}]
}

example_forecast_point_XGBoost = {
    "col_target": "load_consumption",
    "evaluation_index": 7,
    "last_know_index": 9,
    "lag": 1,
    "type": "predictions",
    "norm_values": "False",
    "model_architecture_params": {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "json_list_df_all_data_norm": [
        {"load_consumption": 0.6800409376, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7391304348, "minute": 0.8983050847, "second": 0, "hour_sin": -0.9659258263,
         "hour_cos": -0.2588190451, "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934,
         "week_sin": -0.7485107482, "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6952882419, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7391304348, "minute": 0.9830508475, "second": 0, "hour_sin": -0.9659258263,
         "hour_cos": -0.2588190451, "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934,
         "week_sin": -0.7485107482, "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.7110373283, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.0508474576, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6988007163, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.1355932203, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6925315077, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.2203389831, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6344474735, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.3050847458, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5210668107, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.3898305085, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5220188471, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.4745762712, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5324090483, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.5593220339, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5249571553, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.6440677966, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0}]
}

example_forecast_neural_networks = {
    "col_target": "load_consumption",
    "evaluation_index": 7,
    "last_know_index": 9,
    "type": "predictions",
    "norm_values": "True",
    "model_architecture_params": [{
        "layers": {
            "layer_0": {
                "model_type": "LSTM",
                "neurons": 3,
                "recurrent_dropout": 0,
                "activation": "relu",
                "l2_regularizers": 0
            },
            "layer_1": {
                "model_type": "Bi-LSTM",
                "neurons": 1,
                "recurrent_dropout": 0,
                "activation": "relu",
                "l2_regularizers": 0
            },
            "layer_2": {
                "model_type": "Bi-LSTM",
                "neurons": 1,
                "recurrent_dropout": 0,
                "activation": "relu",
                "l2_regularizers": 0
            }
        },
        "epochs": 5,
        "optimizer": "adam",
        "lag": 4,
        "points_per_call": 4,
        "final_l2_regularizer": 0.2,
        "activation": "linear"
    }],
    "json_list_df_all_data_norm": [
        {"load_consumption": 0.6800409376, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7391304348, "minute": 0.8983050847, "second": 0, "hour_sin": -0.9659258263,
         "hour_cos": -0.2588190451, "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934,
         "week_sin": -0.7485107482, "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6952882419, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7391304348, "minute": 0.9830508475, "second": 0, "hour_sin": -0.9659258263,
         "hour_cos": -0.2588190451, "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934,
         "week_sin": -0.7485107482, "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.7110373283, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.0508474576, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6988007163, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.1355932203, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6925315077, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.2203389831, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.6344474735, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.3050847458, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5210668107, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.3898305085, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5220188471, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.4745762712, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5324090483, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.5593220339, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0},
        {"load_consumption": 0.5249571553, "year": 0.984, "week": 0.6274509804, "day_of_week": 0.8333333333,
         "hour": 0.7826086957, "minute": 0.6440677966, "second": 0, "hour_sin": -1, "hour_cos": -1.836970199e-16,
         "day_of_week_sin": -0.9749279122, "day_of_week_cos": -0.222520934, "week_sin": -0.7485107482,
         "week_cos": -0.6631226582, "is_holiday": 0}]
}

example_metrix_all = {
    "col_time": "time",
    "col_target": "load_consumption",
    "json_list_df_reverse_evaluation": [
        {"load_consumption": 18256.488175171136, "time": "2023-12-17 00:07:00"},
        {"load_consumption": 18557.543205893933, "time": "2023-12-17 00:12:00"},
        {"load_consumption": 18732.26521594262, "time": "2023-12-17 00:17:00"},
        {"load_consumption": 18786.505777776718, "time": "2023-12-17 00:22:00"},
        {"load_consumption": 18778.05954055041, "time": "2023-12-17 00:27:00"},
        {"load_consumption": 18707.63702002251, "time": "2023-12-17 00:32:00"},
        {"load_consumption": 18585.811475117684, "time": "2023-12-17 00:37:00"},
        {"load_consumption": 18423.762592223524, "time": "2023-12-17 00:42:00"},
        {"load_consumption": 18245.625279249252, "time": "2023-12-17 00:47:00"},
        {"load_consumption": 18069.144327852846, "time": "2023-12-17 00:52:00"}
    ],
    "json_list_df_reverse_comparative": [
        {"load_consumption": 17574.4, "time": "2023-12-17 00:02:00"},
        {"load_consumption": 17930, "time": "2023-12-17 00:07:00"},
        {"load_consumption": 16916.699999999997, "time": "2023-12-17 00:12:00"},
        {"load_consumption": 18008.1, "time": "2023-12-17 00:17:00"},
        {"load_consumption": 17515.1, "time": "2023-12-17 00:22:00"},
        {"load_consumption": 19243.9, "time": "2023-12-17 00:27:00"},
        {"load_consumption": 15745.6, "time": "2023-12-17 00:32:00"},
        {"load_consumption": 16434.9, "time": "2023-12-17 00:37:00"},
        {"load_consumption": 16154.700000000003, "time": "2023-12-17 00:42:00"},
        {"load_consumption": 15885.500000000002, "time": "2023-12-17 00:47:00"}
    ],
}

example_forecast_prophet = {
    "col_target": "load_consumption",
    "col_time": "time",
    "evaluation_index": 7,
    "last_know_index": 9,
    "count_forecast_point": 2,
    "type": "predictions",
    "json_list_df_all_data": [
        {
            "time": "2023-12-18 07:51:18",
            "load_consumption": 30035.7
        },
        {
            "time": "2023-12-18 07:56:19",
            "load_consumption": 30388.8
        },
        {
            "time": "2023-12-18 08:01:18",
            "load_consumption": 28574.1
        },
        {
            "time": "2023-12-18 08:06:18",
            "load_consumption": 28788.9
        },
        {
            "time": "2023-12-18 08:11:17",
            "load_consumption": 30401.4
        },
        {
            "time": "2023-12-18 08:16:17",
            "load_consumption": 38862.6
        },
        {
            "time": "2023-12-18 08:21:18",
            "load_consumption": 47326.1
        },
        {
            "time": "2023-12-18 08:26:18",
            "load_consumption": 43447.9
        },
        {
            "time": "2023-12-18 08:31:17",
            "load_consumption": 42628.3
        },
        {
            "time": "2023-12-18 08:36:17",
            "load_consumption": 45304.2
        },
        {
            "time": "2023-12-18 08:41:17",
            "load_consumption": 44403.9
        },
        {
            "time": "2023-12-18 08:46:17",
            "load_consumption": 45483.5
        },
        {
            "time": "2023-12-18 08:51:16",
            "load_consumption": 47645.5
        },
        {
            "time": "2023-12-18 08:56:16",
            "load_consumption": 45234.6
        },
        {
            "time": "2023-12-18 09:01:17",
            "load_consumption": 45466.1
        },
        {
            "time": "2023-12-18 09:06:17",
            "load_consumption": 47131.2
        },
        {
            "time": "2023-12-18 09:11:16",
            "load_consumption": 45947.5
        },
        {
            "time": "2023-12-18 09:16:16",
            "load_consumption": 45396.9
        },
        {
            "time": "2023-12-18 09:21:16",
            "load_consumption": 45587.1
        },
        {
            "time": "2023-12-18 09:26:16",
            "load_consumption": 46904.2
        }
    ]
}

