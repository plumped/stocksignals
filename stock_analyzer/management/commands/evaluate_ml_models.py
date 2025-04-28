# stock_analyzer/management/commands/evaluate_ml_models.py
from django.core.management.base import BaseCommand
from stock_analyzer.ml_evaluate import evaluate_all_models
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Bewertet alle ML-Modelle und erstellt Berichte'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            type=str,
            default='ml_evaluation',
            help='Verzeichnis für Bewertungsberichte',
        )

    def handle(self, *args, **options):
        output_dir = options['output_dir']

        self.stdout.write(f'Starte ML-Modellbewertung um {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        try:
            results = evaluate_all_models(output_dir)

            if results:
                model_count = len(results)
                avg_direction_accuracy = sum(
                    result.get('price_direction_accuracy', 0)
                    for result in results.values() if 'price_direction_accuracy' in result
                ) / model_count if model_count > 0 else 0

                avg_signal_accuracy = sum(
                    result.get('signal_accuracy', 0)
                    for result in results.values() if 'signal_accuracy' in result
                ) / model_count if model_count > 0 else 0

                self.stdout.write(
                    self.style.SUCCESS(
                        f'Bewertung abgeschlossen für {model_count} Modelle\n'
                        f'Durchschnittliche Richtungsgenauigkeit: {avg_direction_accuracy:.2%}\n'
                        f'Durchschnittliche Signalgenauigkeit: {avg_signal_accuracy:.2%}'
                    )
                )

                self.stdout.write(f'Berichte wurden im Verzeichnis "{output_dir}" erstellt')
            else:
                self.stdout.write(
                    self.style.WARNING('Keine Modelle zur Bewertung gefunden')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Fehler bei der Modellbewertung: {str(e)}')
            )
            logger.error(f'Fehler bei der Modellbewertung: {str(e)}')