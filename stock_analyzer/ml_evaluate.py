# stock_analyzer/ml_evaluate.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, roc_curve, auc
import logging

from .models import Stock, StockData
from .ml_models import MLPredictor

logger = logging.getLogger(__name__)


def evaluate_all_models(output_dir='ml_evaluation'):
    """Evaluiert alle ML-Modelle und generiert Berichte"""

    # Ausgabeverzeichnis erstellen, falls nicht vorhanden
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Alle Aktien mit ML-Modellen finden
    model_files = []
    ml_models_dir = 'ml_models'

    if os.path.exists(ml_models_dir):
        model_files = [f for f in os.listdir(ml_models_dir) if f.endswith('_price_model.pkl')]

    results = {}

    for model_file in model_files:
        symbol = model_file.replace('_price_model.pkl', '')
        try:
            # ML-Predictor erstellen
            predictor = MLPredictor(symbol)

            # Modell evaluieren
            performance = predictor.evaluate_model_performance()

            if performance:
                results[symbol] = performance

                # Visualisierung erstellen (optional)
                if 'classification_report' in performance:
                    _create_model_visualization(symbol, performance, output_dir)

                logger.info(f"Modell für {symbol} erfolgreich evaluiert")
            else:
                logger.warning(f"Keine Performance-Daten für {symbol}")

        except Exception as e:
            logger.error(f"Fehler bei der Evaluation von {symbol}: {str(e)}")

    # Zusammenfassenden Bericht erstellen
    _create_summary_report(results, output_dir)

    return results


def _create_model_visualization(symbol, performance, output_dir):
    """Erstellt Visualisierungen für die Modellperformance"""
    try:
        # Confusion Matrix für Signal-Modell visualisieren
        if 'classification_report' in performance:
            report = performance['classification_report']

            # Confusion Matrix aus dem Report extrahieren (falls vorhanden)
            if hasattr(report, 'confusion_matrix'):
                cm = report.confusion_matrix

                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f'Confusion Matrix - {symbol}')
                plt.colorbar()

                classes = ['Sell', 'Hold', 'Buy']
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes)
                plt.yticks(tick_marks, classes)

                plt.xlabel('Predicted')
                plt.ylabel('True')

                # Werte in die Matrix eintragen
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{symbol}_confusion_matrix.png'))
                plt.close()

        # Feature Importance visualisieren, falls vorhanden
        if 'feature_importance' in performance:
            importance = performance['feature_importance']

            # Top 10 Features auswählen
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, values = zip(*sorted_features)

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), values, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title(f'Top Feature Importance - {symbol}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{symbol}_feature_importance.png'))
            plt.close()

    except Exception as e:
        logger.error(f"Fehler bei der Visualisierungserstellung für {symbol}: {str(e)}")


def _create_summary_report(results, output_dir):
    """Erstellt einen zusammenfassenden Bericht über alle Modelle"""
    try:
        # Prüfen, ob wir tatsächlich Ergebnisse haben
        if not results or len(results) == 0:
            logger.warning("Keine Ergebnisse für Zusammenfassungsbericht vorhanden")
            return

        # Zusammenfassung in Dataframe konvertieren
        summary_data = []

        for symbol, perf in results.items():
            summary_row = {
                'Symbol': symbol,
                'Price_RMSE': perf.get('price_rmse', np.nan),
                'Direction_Accuracy': perf.get('price_direction_accuracy',
                                               np.nan) * 100 if 'price_direction_accuracy' in perf else np.nan,
                'Signal_Accuracy': perf.get('signal_accuracy', np.nan) * 100 if 'signal_accuracy' in perf else np.nan
            }
            summary_data.append(summary_row)

        # In Dataframe umwandeln
        df_summary = pd.DataFrame(summary_data)

        # Prüfen, ob der Dataframe leer ist oder keine Daten enthält
        if df_summary.empty or df_summary['Direction_Accuracy'].isna().all():
            logger.warning("Keine gültigen Daten für den Zusammenfassungsbericht gefunden")
            return

        # Nach Direction Accuracy sortieren, falls vorhanden
        if 'Direction_Accuracy' in df_summary.columns and not df_summary['Direction_Accuracy'].isna().all():
            df_summary = df_summary.sort_values('Direction_Accuracy', ascending=False)

        # Als CSV speichern
        csv_path = os.path.join(output_dir, 'model_performance_summary.csv')
        df_summary.to_csv(csv_path, index=False)

        # Visualisierung der Genauigkeit über alle Modelle
        plt.figure(figsize=(12, 8))

        # Direction Accuracy Plot
        plt.subplot(2, 1, 1)
        if not df_summary['Direction_Accuracy'].isna().all():
            valid_data = df_summary.dropna(subset=['Direction_Accuracy'])
            plt.bar(valid_data['Symbol'], valid_data['Direction_Accuracy'], color='skyblue')
            plt.axhline(y=valid_data['Direction_Accuracy'].mean(), color='r', linestyle='-', label='Durchschnitt')
            plt.title('Richtungs-Genauigkeit nach Aktie (%)')
            plt.xticks(rotation=90)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center')
            plt.title('Richtungs-Genauigkeit (keine Daten)')

        # Signal Accuracy Plot
        plt.subplot(2, 1, 2)
        if not df_summary['Signal_Accuracy'].isna().all():
            valid_data = df_summary.dropna(subset=['Signal_Accuracy'])
            plt.bar(valid_data['Symbol'], valid_data['Signal_Accuracy'], color='lightgreen')
            plt.axhline(y=valid_data['Signal_Accuracy'].mean(), color='r', linestyle='-', label='Durchschnitt')
            plt.title('Signal-Genauigkeit nach Aktie (%)')
            plt.xticks(rotation=90)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center')
            plt.title('Signal-Genauigkeit (keine Daten)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_summary.png'))
        plt.close()

        # Textbericht erstellen
        with open(os.path.join(output_dir, 'model_performance_report.txt'), 'w') as f:
            f.write("ML-Modell Performance Zusammenfassung\n")
            f.write("===================================\n\n")
            f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Anzahl der evaluierten Modelle: {len(results)}\n")

            if not df_summary['Direction_Accuracy'].isna().all():
                f.write(f"Durchschnittliche Richtungs-Genauigkeit: {df_summary['Direction_Accuracy'].mean():.2f}%\n")
            else:
                f.write("Durchschnittliche Richtungs-Genauigkeit: Keine Daten\n")

            if not df_summary['Signal_Accuracy'].isna().all():
                f.write(f"Durchschnittliche Signal-Genauigkeit: {df_summary['Signal_Accuracy'].mean():.2f}%\n\n")
            else:
                f.write("Durchschnittliche Signal-Genauigkeit: Keine Daten\n\n")

            if not df_summary.empty and not df_summary['Direction_Accuracy'].isna().all():
                f.write("Top 5 Modelle nach Richtungs-Genauigkeit:\n")
                top_models = df_summary.sort_values('Direction_Accuracy', ascending=False).head(5)
                for idx, row in top_models.iterrows():
                    f.write(f"  - {row['Symbol']}: {row['Direction_Accuracy']:.2f}%\n")
            else:
                f.write("Keine ausreichenden Daten für Top-Modelle vorhanden\n")

            f.write("\nWeitere Details in 'model_performance_summary.csv'\n")

        logger.info(f"Zusammenfassender Bericht wurde im Verzeichnis '{output_dir}' erstellt")

    except Exception as e:
        logger.error(f"Fehler beim Erstellen des zusammenfassenden Berichts: {str(e)}")
        # Hier sollten wir keinen Fehler werfen, sondern nur im Log dokumentieren, um den Prozess nicht zu unterbrechen