# management/commands/generate_ielts_questions.py

import hashlib
import os
import random
import time

import requests
from django.core.management.base import BaseCommand
from django.db import transaction

from bot.models import Part, Question, Test


class Command(BaseCommand):
    help = "Generate IELTS speaking test questions using Hugging Face API"

    PART1_TOPICS = {
        "Personal Information": ["name", "hometown", "family", "work", "studies"],
        "Home": ["accommodation", "neighborhood", "rooms", "decoration", "furniture"],
        "Daily Life": ["routine", "free time", "weekends", "meals", "shopping"],
        "Entertainment": ["movies", "music", "reading", "TV shows", "social media"],
        "Nature": ["weather", "seasons", "plants", "animals", "environmental issues"],
        "Transportation": [
            "public transport",
            "driving",
            "cycling",
            "travel",
            "traffic",
        ],
        "Health": ["exercise", "diet", "sleep", "sports", "lifestyle"],
        "Technology": ["internet", "smartphones", "computers", "apps", "gadgets"],
    }

    PART2_TOPICS = {
        "People": ["friend", "family member", "teacher", "neighbor", "role model"],
        "Places": ["city", "restaurant", "park", "building", "tourist spot"],
        "Objects": ["gift", "device", "clothing", "artwork", "photograph"],
        "Events": ["celebration", "trip", "achievement", "meeting", "performance"],
        "Experiences": [
            "learning experience",
            "challenge",
            "success",
            "mistake",
            "discovery",
        ],
        "Activities": ["hobby", "skill", "sport", "project", "routine"],
    }

    PART3_TOPICS = {
        "Society": ["education", "work", "family", "culture", "tradition"],
        "Technology": [
            "innovation",
            "social media",
            "automation",
            "privacy",
            "future tech",
        ],
        "Environment": [
            "climate change",
            "sustainability",
            "urbanization",
            "conservation",
        ],
        "Lifestyle": [
            "work-life balance",
            "health",
            "entertainment",
            "social connections",
        ],
        "Global Issues": ["economy", "migration", "development", "globalization"],
    }
    def is_question_unique(self, question_text):
        """Check if question is unique using content hash"""
        content_hash = hashlib.md5(question_text.lower().encode()).hexdigest()
        return not Question.objects.filter(content_hash=content_hash).exists()

    def add_arguments(self, parser):
        parser.add_argument("num_tests", type=int, help="Number of tests to generate")

    def handle(self, *args, **kwargs):
        num_tests = kwargs["num_tests"]
        self.stdout.write(f"Generating {num_tests} IELTS tests...")

        for test_num in range(1, num_tests + 1):
            try:
                with transaction.atomic():
                    self.generate_test(test_num)
                    time.sleep(2)  # Avoid rate limiting
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"Error generating Test {test_num}: {str(e)}")
                )


    def generate_test(self, test_num):
        self.stdout.write(f"Generating Test {test_num}...")

        # Create Test
        test = Test.objects.create(number=test_num, title=f"Test {test_num}")

        # Generate questions for each part
        for part_num in range(1, 4):
            part = Part.objects.create(test=test, number=part_num)

            if part_num == 1:
                self.generate_part1_questions(part)
            elif part_num == 2:
                self.generate_part2_questions(part)
            else:
                self.generate_part3_questions(part)

    def call_huggingface_api(self, prompt):
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

        try:
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 500}},
            )
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"API Error: {str(e)}"))
            return None

    def generate_part1_questions(self, part):
        used_topics = set()
        for category, topics in self.PART1_TOPICS.items():
            if len(used_topics) >= 4:  # Ensure variety across categories
                break

            prompt = f"""Generate 1-2 unique IELTS Speaking Part 1 questions about {category.lower()}, specifically about {", ".join(topics)}.
The questions should be simple, direct, and conversational.
Format: One question per line, starting with a number.
Example: 1. What kind of... 2. How often do you..."""

            response = self.call_huggingface_api(prompt)
            if response:
                questions = [
                    q.strip()
                    for q in response.split("\n")
                    if q.strip() and any(c.isdigit() for c in q)
                ]

                for i, question in enumerate(questions):
                    clean_question = " ".join(question.split()[1:])
                    if self.is_question_unique(clean_question):
                        Question.objects.create(
                            part=part,
                            text=clean_question,
                            category=category,
                            content_hash=hashlib.md5(
                                clean_question.lower().encode()
                            ).hexdigest(),
                            order=len(used_topics),
                        )
                        used_topics.add(clean_question)

    def generate_part2_questions(self, part):
        for category, topics in self.PART2_TOPICS.items():
            topic = random.choice(topics)

            prompt = f"""Generate an IELTS Speaking Part 2 cue card topic about {topic}.
Format:
Main question: Describe a [topic]...
You should say:
• [point 1]
• [point 2]
• [point 3]
• and explain [point 4]"""

            response = self.call_huggingface_api(prompt)
            if response and self.is_question_unique(response):
                lines = response.split("\n")
                main_question = lines[0].strip()
                cue_points = "\n".join(
                    line.strip() for line in lines[1:] if line.strip()
                )

                Question.objects.create(
                    part=part,
                    text=main_question,
                    cue_card_points=cue_points,
                    category=category,
                    content_hash=hashlib.md5(response.lower().encode()).hexdigest(),
                    order=0,
                )
                break

    def generate_part3_questions(self, part):
        used_topics = set()
        for category, topics in self.PART3_TOPICS.items():
            if len(used_topics) >= 4:
                break

            prompt = f"""Generate 1-2 analytical IELTS Speaking Part 3 questions about {category.lower()}, focusing on {", ".join(topics)}.
Questions should require in-depth discussion of trends, causes, effects, or future implications.
Format: One question per line.
Example: How might [topic] change in the future? What are the main reasons for [topic]?"""

            response = self.call_huggingface_api(prompt)
            if response:
                questions = [
                    q.strip()
                    for q in response.split("\n")
                    if q.strip() and any(c.isdigit() for c in q)
                ]

                for question in questions:
                    clean_question = " ".join(question.split()[1:])
                    if self.is_question_unique(clean_question):
                        Question.objects.create(
                            part=part,
                            text=clean_question,
                            category=category,
                            content_hash=hashlib.md5(
                                clean_question.lower().encode()
                            ).hexdigest(),
                            order=len(used_topics),
                        )
                        used_topics.add(clean_question)
