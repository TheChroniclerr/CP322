import pandas as pd
from preprocessing_data.pre_processing import preprocessing
from feature_engineering.build_features import build
from models.train_model import train
from models.predict_model import predict
from visualization.visualize import visualize


RAW_PATH = "src/data/raw/usnews_data.txt"
PROCESSED_PATH = "src/data/processed/usnews_cleaned.csv"
FIGURE_PATH = "reports/figures/"

figure_names = [
    "correlation_heatmap.png",
    "knn_hptuning.png",
    "rf_hptuning.png",
    "perf_comp_bar.png",
    "graduation_rate_dist.png",
    "lr_feature_effects.png",
    "lr_avp.png",
    "lr_residual.png",
    "knn_avp.png",
    "knn_residual.png",
    "rf_feature_importance.png",
    "rf_avp.png",
    "rf_residual.png"
]

if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    df = preprocessing(df)
    df = build(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(PROCESSED_PATH)

    models, unscaled, scaled = train(df)
    predictions = predict(models, unscaled, scaled)
    
    figures = visualize(df, models, unscaled, scaled, predictions)
    for name in figure_names:
        path = FIGURE_PATH + name
        figures.figure(name)
        figures.savefig(path, bbox_inches='tight')
        print(path)
    figures.close()