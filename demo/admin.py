from django.contrib import admin

from .models import QuestionAnswer


class QuestionAnswerAdmin(admin.ModelAdmin):
    list_display = ('img_url', 'question', 'top5_answer', 'created_at', 'socketid')

admin.site.register(QuestionAnswer, QuestionAnswerAdmin)
