from django.contrib import admin

from . import models


@admin.register(models.User)
class UserAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "telegram_id",
        "username",
        "first_name",
        "last_name",
        "created_at",
    ]
    search_fields = ["telegram_id", "username", "first_name", "last_name"]


@admin.register(models.Attempt)
class AttemptAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "created_at"]
    search_fields = [
        "user__telegram_id",
        "user__username",
        "user__first_name",
        "user__last_name",
    ]
    list_filter = ["created_at"]


@admin.register(models.Test)
class TestAdmin(admin.ModelAdmin):
    list_display = ["number", "title", "created_at"]
    search_fields = ["number", "title"]
    list_filter = ["created_at"]
    ordering = ["number"]


@admin.register(models.Part)
class PartAdmin(admin.ModelAdmin):
    list_display = ["test", "number", "description"]
    search_fields = ["test__number", "test__title", "number", "description"]
    list_filter = ["test"]
    ordering = ["test", "number"]


@admin.register(models.Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ["part", "order", "text"]
    search_fields = ["part__test__number", "part__number", "number", "question"]
    list_filter = ["part"]
    ordering = ["part"]
