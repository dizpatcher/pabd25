import os
import datetime
import joblib
import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import cianparser

from helpers import drop_outliers


logger = None


def setup_logger(log_path="/logs/model.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Проверяем, есть ли уже обработчики (чтобы не дублировать)
    if not logger.handlers:
        handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def parse_cian():
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    data = []
    for n_rooms in range(1, 4):

        new_data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 7,
                "end_page": 9,
                "object_type": "secondary",
            },
        )
        data.extend(new_data)

    df = pd.DataFrame(data)

    csv_path = f"/data/raw/flats_{t}.csv"
    df.to_csv(csv_path, encoding="utf-8", index=False)


def process_data():
    data_path = "/data/raw"
    all_raw_data_files = os.listdir(data_path)
    # Фильтруем только те, которые соответствуют шаблону
    csv_files = [
        f for f in all_raw_data_files if f.startswith("flats_") and f.endswith(".csv")
    ]
    # Чтение и объединение всех файлов
    df = pd.concat(
        [pd.read_csv(f"{data_path}/{f}") for f in csv_files], ignore_index=True
    )
    logger.info(f"Объём всех сырых данных: {len(df)} строк")

    df = df.drop_duplicates(subset=["url"])
    df["flat_id"] = df["url"].str.extract(r"/flat/(\d+)/")[0].astype("int64")
    df.drop(
        [
            "location",
            "deal_type",
            "accommodation_type",
            "price_per_month",
            "commissions",
            "url",
            "author",
            "author_type",
            "residential_complex",
        ],
        axis=1,
        inplace=True,
    )
    df = df.dropna()

    need_columns = ["total_meters", "price"]
    for column in need_columns:
        df = drop_outliers(df, column)

    logger.info(f"Объём данных после обработки: {len(df)} строк")

    test_size = int(len(df["flat_id"]) * 0.2)
    test_items = sorted(df["flat_id"])[-test_size:]

    train_data = df[~df["flat_id"].isin(test_items)]
    test_data = df[df["flat_id"].isin(test_items)]

    return train_data, test_data


def train_model(train_data):
    X_train = train_data[["total_meters"]]  # только один признак - площадь
    y_train = train_data["price"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def test_model(test_data, model):
    X_test = test_data[["total_meters"]]  # только один признак - площадь
    y_test = test_data["price"]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Логирование метрик
    logger.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logger.info(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    logger.info(f"Коэффициент детерминации R²: {r2:.6f}")
    logger.info(
        f"Средняя ошибка предсказания: {np.mean(np.abs(y_test - y_pred)):.2f} рублей"
    )


def save_model(model_name, model):
    model_path = f"/models/{model_name}.pkl"

    joblib.dump(model, model_path)
    logger.info(f"Модель сохранена в файл {model_path}")


def main():
    global logger
    logger = setup_logger()

    logger.info("Запуск скрипта")

    parser = argparse.ArgumentParser(
        description="Обучение и сохранение модели предсказания цен на недвижимость."
    )
    parser.add_argument("--mname", type=str, required=True, help="Model name")
    parser.add_argument("--mversion", type=str, required=True, help="Model version")

    args = parser.parse_args()
    model_name = args.mname
    model_version = args.mversion

    parse_cian()
    train_data, test_data = process_data()

    model = train_model(train_data)
    test_model(test_data, model)

    save_model(f"{model_name}_model_{model_version}", model)


if __name__ == "__main__":
    main()
