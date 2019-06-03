from __future__ import unicode_literals

from django.db import models
from django.utils.html import format_html


class QuestionAnswer(models.Model):
    top5_answer = models.CharField(max_length=10000, blank=True, null=True)
    image = models.CharField(max_length=1000, blank=True, null=True)
    question = models.CharField(max_length=10000, blank=True, null=True)
    created_at = models.DateTimeField("Time", null=True, auto_now_add=True)
    socketid = models.CharField(max_length=10000, blank=True, null=True)
    vqa_model = models.CharField(max_length=10000, blank=True, null=True)

    def __unicode__(self):
        return self.question

    def img_url(self):
        return format_html("<img src='{}' height='150px'>", self.image)
