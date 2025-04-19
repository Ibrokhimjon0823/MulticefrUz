from django.contrib.auth.models import AbstractUser
from django.db import models

from .managers import UserManager


class User(AbstractUser):
    telegram_id = models.BigIntegerField(unique=True)
    username = models.CharField(max_length=255, null=True, blank=True)
    first_name = models.CharField(max_length=255, null=True, blank=True)
    last_name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    USERNAME_FIELD = "telegram_id"
    REQUIRED_FIELDS = []

    objects = UserManager()

    def __str__(self):
        return f"{self.telegram_id} - {self.username if self.username else self.first_name}"


class Attempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="attempts")
    transcript = models.TextField()
    evaluation = models.TextField()
    score = models.FloatField(null=True, blank=True)  # Score out of 9
    feedback = models.TextField(null=True, blank=True)  # Feedback from the evaluator
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Attempt {self.id} by {self.user}"


class Test(models.Model):
    number = models.IntegerField(unique=True)
    title = models.CharField(max_length=100)  # e.g., "Test 1"
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Test {self.number}"

    class Meta:
        ordering = ["number"]


class Part(models.Model):
    PART_CHOICES = [
        (1, "Part 1"),
        (2, "Part 2"),
        (3, "Part 3"),
    ]

    test = models.ForeignKey(Test, on_delete=models.CASCADE, related_name="parts")
    number = models.IntegerField(choices=PART_CHOICES)
    description = models.TextField(blank=True)  # Optional description of this part

    class Meta:
        unique_together = ["test", "number"]
        ordering = ["number"]

    def __str__(self):
        return f"{self.test} - Part {self.number}"


class Question(models.Model):
    part = models.ForeignKey(Part, on_delete=models.CASCADE, related_name="questions")
    text = models.TextField()
    cue_card_points = models.TextField(blank=True)  # For Part 2 bullet points
    category = models.CharField(
        max_length=50, default="IELTS"
    )  # Store the topic category
    content_hash = models.CharField(
        max_length=32, default="default_hash"
    )  # For uniqueness check
    order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"{self.part} - Question {self.order}"
