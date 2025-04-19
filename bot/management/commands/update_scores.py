import re

from django.core.management.base import BaseCommand
from django.db import transaction

from bot.models import Attempt


class Command(BaseCommand):
    help = "Extract overall band scores from evaluation text and save to score field"

    def handle(self, *args, **options):
        attempts = Attempt.objects.all()
        updated_count = 0

        with transaction.atomic():
            for attempt in attempts:
                if attempt.evaluation:
                    # Use regex to find the overall band score pattern
                    score_pattern = r"\[Overall Band Score\]:\s*(\d+\.\d+)\/9"
                    match = re.search(score_pattern, attempt.evaluation)

                    if match:
                        score = float(match.group(1))
                        attempt.score = score
                        attempt.save(update_fields=["score"])
                        updated_count += 1
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"Updated Attempt {attempt.id} with score {score}"
                            )
                        )
                    else:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Could not find score in Attempt {attempt.id}"
                            )
                        )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Attempt {attempt.id} has no evaluation text"
                        )
                    )

            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully updated {updated_count} out of {attempts.count()} attempts"
                )
            )
