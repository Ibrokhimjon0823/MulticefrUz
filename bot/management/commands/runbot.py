import math
import os
import re
from typing import Dict, List

import requests
import stanza
import whisper
from django.core.management.base import BaseCommand
from pydub import AudioSegment
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    Filters,
    MessageHandler,
    Updater,
)

from bot.models import Attempt, Part, Question, Test, User

# Constants
MAX_AUDIO_DURATION = 120  # 2 minutes in seconds
WHISPER_MODEL = whisper.load_model("base")
HF_API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
)
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}


def get_or_create_user(update: Update):
    """Create or update user in database"""
    user_data = update.effective_user
    user, created = User.objects.get_or_create(
        telegram_id=user_data.id,
        defaults={
            "username": user_data.username,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
        },
    )
    return user


def show_main_menu(update: Update, context: CallbackContext):
    """Show main menu with inline keyboard"""
    user = get_or_create_user(update)
    welcome_msg = f"Welcome {user.first_name}!\nChoose an option:"

    keyboard = [
        [
            InlineKeyboardButton("📝 Practice", callback_data="select_test"),
            InlineKeyboardButton("📚 History", callback_data="history"),
        ],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        update.callback_query.edit_message_text(
            text=welcome_msg, reply_markup=reply_markup
        )
    else:
        update.message.reply_text(text=welcome_msg, reply_markup=reply_markup)


def start(update: Update, context: CallbackContext):
    """Enhanced start command with persistent keyboard"""
    get_or_create_user(update)

    # Persistent reply keyboard
    reply_keyboard = [["/start", "/help"], ["Practice Questions", "My History"]]
    reply_markup = ReplyKeyboardMarkup(
        reply_keyboard, resize_keyboard=True, persistent=True
    )

    update.message.reply_text(
        "Welcome back! Use menu below:", reply_markup=reply_markup
    )
    show_main_menu(update, context)


def handle_callback(update: Update, context: CallbackContext):
    """Handle callback queries from inline keyboards"""
    query = update.callback_query

    # Acknowledge query immediately to avoid timeout
    query.answer()

    # Process callback data
    if query.data == "start":
        show_main_menu(update, context)
    elif query.data == "select_test":
        show_test_selection(update, context)
    elif query.data.startswith("test_"):
        test_num = query.data.split("_")[1]
        show_test_parts(update, context, test_num)
    elif query.data.startswith("part_"):
        _, test_num, part_num = query.data.split("_")
        show_questions(update, context, test_num, part_num)
    elif query.data.startswith("q_"):
        _, test_num, part_num, q_num = query.data.split("_")
        show_selected_question(update, context, test_num, part_num, q_num)
    elif query.data == "history":
        show_history(update, context)
    elif query.data == "help":
        show_help(update, context)


def show_help(update: Update, context: CallbackContext):
    """Show help information about how to use the bot"""
    help_text = (
        "*Welcome to IELTS Speaking Practice Bot!* 🎯\n\n"
        "*How to Use:*\n"
        "1️⃣ Select 'Practice Questions' from the menu\n"
        "2️⃣ Choose a Test (1-3)\n"
        "3️⃣ Select a Part (1-3)\n"
        "4️⃣ Pick a question to answer\n"
        "5️⃣ Record your voice message (max 2 minutes)\n\n"
        "*About IELTS Speaking Parts:*\n"
        "• *Part 1*: Simple questions about familiar topics\n"
        "• *Part 2*: Long-turn speaking with cue card\n"
        "• *Part 3*: In-depth discussion questions\n\n"
        "*Evaluation:*\n"
        "You'll receive feedback on:\n"
        "📊 Fluency & Coherence\n"
        "📚 Lexical Resource\n"
        "🔤 Grammatical Range\n"
        "🗣 Pronunciation\n\n"
        "*Need Help?*\n"
        "Contact: @your_support_username"
    )

    keyboard = [[InlineKeyboardButton("« Back to Menu", callback_data="start")]]

    if update.callback_query:
        update.callback_query.edit_message_text(
            text=help_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )
    else:
        update.message.reply_text(
            text=help_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )


def show_test_parts(update: Update, context: CallbackContext, test_num: str):
    """Show parts of selected test from database"""
    parts = Part.objects.filter(
        test__number=test_num, questions__is_active=True
    ).distinct()

    keyboard = [
        [
            InlineKeyboardButton(
                f"Part {part.number}", callback_data=f"part_{test_num}_{part.number}"
            )
        ]
        for part in parts
    ]
    keyboard.append(
        [InlineKeyboardButton("« Back to Tests", callback_data="select_test")]
    )

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(
        f"Select a part from Test {test_num}:", reply_markup=reply_markup
    )


def show_questions(
    update: Update, context: CallbackContext, test_num: str, part_num: str
):
    """Show questions for selected part from database"""
    questions = Question.objects.filter(
        part__test__number=test_num, part__number=part_num, is_active=True
    ).order_by("order")

    keyboard = [
        [
            InlineKeyboardButton(
                f"Question {idx + 1}",
                callback_data=f"q_{test_num}_{part_num}_{question.id}",
            )
        ]
        for idx, question in enumerate(questions)
    ]
    keyboard.append(
        [InlineKeyboardButton("« Back to Parts", callback_data=f"test_{test_num}")]
    )

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(
        f"Select a question from Part {part_num}:", reply_markup=reply_markup
    )


def show_selected_question(
    update: Update,
    context: CallbackContext,
    test_num: str,
    part_num: str,
    question_id: str,
):
    """Show selected question from database"""
    try:
        question = Question.objects.get(id=question_id, is_active=True)

        message = f"*Your Question*:\n{question.text}\n\n"

        # Add cue card points for Part 2 questions
        if question.part.number == 2 and question.cue_card_points:
            message += f"\n*Cue Card Points*:\n{question.cue_card_points}\n\n"

        message += (
            "🎙 *Instructions*:\n"
            "1. Record your response (max 2 minutes)\n"
            "2. Send the voice message\n"
            "3. Wait for your evaluation\n\n"
            "_Tip: Structure your response and speak naturally_"
        )

        keyboard = [
            [
                InlineKeyboardButton(
                    "« Back to Questions", callback_data=f"part_{test_num}_{part_num}"
                )
            ]
        ]

        update.callback_query.edit_message_text(
            message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"
        )
    except Question.DoesNotExist:
        update.callback_query.edit_message_text(
            "Question not found. Please try another one.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "« Back to Questions",
                            callback_data=f"part_{test_num}_{part_num}",
                        )
                    ]
                ]
            ),
        )


def show_test_selection(update: Update, context: CallbackContext):
    """Show available IELTS tests from database"""
    tests = Test.objects.filter(parts__questions__is_active=True).distinct()

    keyboard = [
        [
            InlineKeyboardButton(
                f"Test {test.number}", callback_data=f"test_{test.number}"
            )
        ]
        for test in tests
    ]
    keyboard.append([InlineKeyboardButton("« Back to Menu", callback_data="start")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text("Select a test:", reply_markup=reply_markup)


def handle_voice(update: Update, context: CallbackContext):
    """Process voice messages with duration check"""
    update.message.reply_text("⏳ Analyzing your response...")

    try:
        # Check audio duration
        if update.message.voice.duration > MAX_AUDIO_DURATION:
            update.message.reply_text(
                f"❌ Audio too long! Maximum allowed is {MAX_AUDIO_DURATION // 60} minutes"
            )
            return

        # Process audio
        user = get_or_create_user(update)
        voice_file = update.message.voice.get_file()

        # Download and convert audio
        ogg_path = f"audio_{user.telegram_id}.ogg"
        wav_path = f"audio_{user.telegram_id}.wav"
        voice_file.download(ogg_path)
        AudioSegment.from_ogg(ogg_path).export(wav_path, format="wav")

        # Transcribe with Whisper
        result = WHISPER_MODEL.transcribe(wav_path)
        transcript = result["text"]

        # Get evaluation from Hugging Face
        raw_evaluation = get_hf_evaluation(transcript)

        # Save attempt to database
        Attempt.objects.create(
            user=user, transcript=transcript, evaluation=raw_evaluation
        )

        # Format response and send message with keyboard
        formatted_response = format_evaluation_response(transcript, raw_evaluation)

        # Create keyboard markup
        keyboard = [
            [InlineKeyboardButton("📚 View History", callback_data="history")],
            [
                InlineKeyboardButton(
                    "🎯 Try Another Question", callback_data="select_test"
                )
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Send message with keyboard
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=formatted_response,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )

    except Exception as e:
        update.message.reply_text(f"🚨 Error processing your request: {str(e)}")
    finally:
        # Cleanup temporary files
        if "ogg_path" in locals():
            os.remove(ogg_path)
        if "wav_path" in locals():
            os.remove(wav_path)


def format_evaluation_response(transcript, evaluation):
    """Format evaluation response without keyboard markup"""
    # Extract scores using regex
    score_pattern = r"\[(.*?)\]:\s*([\d.]+)/9"
    scores = dict(re.findall(score_pattern, evaluation))

    # Calculate overall score
    try:
        categories = [
            "Fluency/Coherence",
            "Lexical Resource",
            "Grammatical Range",
            "Pronunciation",
        ]
        total = sum(float(scores.get(cat, 0)) for cat in categories)
        overall = round((total / 4) * 2) / 2  # Rounds to nearest 0.5
    except:
        overall = "N/A"

    # Clean up feedback section
    feedback = re.split(r"\[Detailed Feedback\]:?", evaluation)[-1].strip()
    feedback = "\n".join(
        [f"• {line.strip()}" for line in feedback.split("\n") if line.strip()]
    )
    tips = re.split(r"\[Tips for improvement\]:?", evaluation)[-1].strip()
    tips = "\n".join([f"• {line.strip()}" for line in tips.split("\n") if line.strip()])

    # Return only the formatted text
    formatted_response = f"""
🎤 *Transcript*:
_{truncate_text(transcript, 300)}_

📊 *Evaluation*:
{format_score("Fluency/Coherence", scores)}
{format_score("Lexical Resource", scores)}
{format_score("Grammatical Range", scores)}
{format_score("Pronunciation", scores)}

🌟 *Overall Band Score*: {validate_score(overall, "Overall Band Score", transcript)}

📝 *Feedback*:
{feedback}

"""
    penalties = calculate_penalties(transcript)
    adjusted_overall = apply_penalties(overall, penalties)

    # Update response with penalty information
    formatted_response += f"\n🔴 *Automatic Penalties Applied:*\n"
    formatted_response += f"- Grammar errors: {penalties['grammar_errors']}\n"
    formatted_response += f"- Word repetitions: {penalties['repetitions']}\n"
    formatted_response += f"- Fillers: {penalties['fillers']}\n"
    formatted_response += f"- Long pauses: {penalties['long_pauses']}\n"
    formatted_response += f"\n🌟 *Adjusted Overall Score*: {validate_score(adjusted_overall, 'Overall Band Score', transcript)}"

    return formatted_response


def get_hf_evaluation(text):
    """Improved evaluation prompt with strict IELTS criteria"""
    prompt = f"""Act as a strict IELTS examiner. Analyze this speaking response using official IELTS band descriptors.
    
**Evaluation Rules:**
1. Score each category 0-9 using .0 or .5 increments ONLY
2. Be critical - common mistakes should lower scores
3. Follow this exact format:

[Fluency/Coherence]: X.5/9
- Hesitations: {{count}}
- Cohesion issues: {{count}}
[Lexical Resource]: X.0/9
- Repetitions: {{count}}
- Inappropriate vocabulary: {{count}}
[Grammatical Range]: X.5/9
- Grammar errors: {{count}}
- Tense errors: {{count}}
[Pronunciation]: X.0/9
- Intelligibility issues: {{count}}
[Overall Band Score]: X.5/9

**Detailed Feedback:**
- List 3 main weaknesses
- Suggest specific improvements

**Response to evaluate:**
"{text}"

Evaluation:"""

    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={
            "inputs": prompt,
            "parameters": {
                "temperature": 0.2,  # More deterministic
                "max_new_tokens": 800,
            },
        },
    )
    print("in get_hf_evaluation")
    return response.json()[0]["generated_text"]


def get_deepseek_evaluation(text):
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": """You are an IELTS speaking exam evaluator. Analyze the provided response considering:
                - Fluency and Coherence
                - Lexical Resource
                - Grammatical Range and Accuracy
                - Pronunciation
                Provide detailed feedback and estimate IELTS band score.""",
            },
            {"role": "user", "content": text},
        ],
        "model": "deepseek-chat",
        "temperature": 0.7,
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "Error getting evaluation. Please try again later."


def get_evaluation(text):
    if os.getenv("TESTING"):
        return get_hf_evaluation(text)
    return get_deepseek_evaluation(text)


def calculate_penalties(transcript: str) -> Dict[str, int]:
    """Automatically detect common errors"""
    return {
        "grammar_errors": len(list(nlp(transcript).iter_errors())),
        "repetitions": len(re.findall(r"\b(\w+)\s+\1\b", transcript, re.I)),
        "fillers": len(re.findall(r"\b(uh|um|like|you know)\b", transcript, re.I)),
        "long_pauses": len(re.findall(r"\.{3,}|…", transcript)),
    }


def apply_penalties(raw_score: float, penalties: dict) -> float:
    """Adjust score based on detected errors"""
    deduction = min(
        (penalties["grammar_errors"] * 0.2)
        + (penalties["repetitions"] * 0.15)
        + (penalties["fillers"] * 0.1)
        + (penalties["long_pauses"] * 0.1),
        2.5,  # Max total deduction
    )
    return max(raw_score - deduction, 0)


def format_score(category, scores):
    score = float(scores.get(category, 0))
    filled = "●" * int(score)
    empty = "○" * (9 - int(score))
    return f"▸ *{category}*: {filled}{empty} {score}/9"


IELTS_BAND_DESCRIPTORS = {
    "9": "Speaks fluently with no hesitation, full grammatical control",
    "8": "Occasional hesitations, wide vocabulary range",
    "7": "Noticeable pauses, some vocabulary limitations",
    "6": "Frequent hesitations, limited vocabulary",
    "5": "Slow speech, simple structures, frequent errors",
}


nlp = stanza.Pipeline(lang="en", processors="tokenize")


def validate_score(score: float, category: str, transcript: str) -> float:
    """Ensure scores match official descriptors"""
    key = str(math.floor(score))
    descriptor = IELTS_BAND_DESCRIPTORS.get(key, "")

    if score >= 7.0 and any(
        [
            "simple structures" in descriptor.lower() in transcript,
            len(nlp(transcript).sentences) < 5,
            len(transcript.split()) < 100,
        ]
    ):
        return min(score, 6.5)
    return score


def truncate_text(text, max_length):
    return (text[:max_length] + "...") if len(text) > max_length else text


def show_history(update: Update, context: CallbackContext):
    """Handle /history command and history callback"""
    user = get_or_create_user(update)
    attempts = Attempt.objects.filter(user=user).order_by("-created_at")[:5]

    if not attempts:
        message = "📚 No previous attempts found!"
    else:
        message = "📖 Last 5 attempts:\n\n"
        for idx, attempt in enumerate(attempts, 1):
            message += (
                f"Attempt #{idx} ({attempt.created_at.date()})\n"
                f"Score: {parse_band_score(attempt.evaluation)}\n"
                f"{'-' * 30}\n"
            )

    keyboard = [[InlineKeyboardButton("« Back to Menu", callback_data="start")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Handle both callback queries and direct commands
    if update.callback_query:
        update.callback_query.edit_message_text(text=message, reply_markup=reply_markup)
    else:
        update.message.reply_text(text=message, reply_markup=reply_markup)


def parse_band_score(evaluation):
    """Extract band score from evaluation text"""
    if "[Overall Band Score]" in evaluation:
        return evaluation.split("[Overall Band Score]:")[-1].split("\n")[0].strip()
    return "N/A"


class Command(BaseCommand):
    help = "Run Telegram Bot"

    def handle(self, *args, **kwargs):
        token = os.getenv("TELEGRAM_TOKEN")
        print("Token:", token)
        if not token:
            self.stdout.write(
                self.style.ERROR("TELEGRAM_TOKEN not found in environment")
            )
            return

        updater = Updater(token=token, use_context=True)  # Add use_context=True
        dp = updater.dispatcher

        # Rest of your code remains the same
        dp.add_handler(CommandHandler("start", start))
        dp.add_handler(CallbackQueryHandler(handle_callback))
        dp.add_handler(MessageHandler(Filters.voice, handle_voice))

        self.stdout.write(self.style.SUCCESS("Bot is running..."))
        updater.start_polling()
        updater.idle()
