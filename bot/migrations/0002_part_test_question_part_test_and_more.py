# Generated by Django 5.1.5 on 2025-01-30 15:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("bot", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Part",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "number",
                    models.IntegerField(
                        choices=[(1, "Part 1"), (2, "Part 2"), (3, "Part 3")]
                    ),
                ),
                ("description", models.TextField(blank=True)),
            ],
            options={
                "ordering": ["number"],
            },
        ),
        migrations.CreateModel(
            name="Test",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("number", models.IntegerField(unique=True)),
                ("title", models.CharField(max_length=100)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["number"],
            },
        ),
        migrations.CreateModel(
            name="Question",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField()),
                ("cue_card_points", models.TextField(blank=True)),
                ("order", models.IntegerField(default=0)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "part",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="questions",
                        to="bot.part",
                    ),
                ),
            ],
            options={
                "ordering": ["order"],
            },
        ),
        migrations.AddField(
            model_name="part",
            name="test",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="parts",
                to="bot.test",
            ),
        ),
        migrations.AlterUniqueTogether(
            name="part",
            unique_together={("test", "number")},
        ),
    ]
