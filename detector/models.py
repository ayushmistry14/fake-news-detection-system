from django.conf import settings
from django.db import models


class PredictionHistory(models.Model):

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='prediction_histories',
        null=True,
        blank=True
    )

    news_title = models.CharField(
        max_length=250,
        blank=True,
        null=True
    )

    news_text = models.TextField()

    prediction = models.CharField(
        max_length=50
    )

    confidence = models.FloatField(
        default=0.0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:

        ordering = ['-created_at']

        verbose_name = 'Prediction History'

        verbose_name_plural = 'Prediction Histories'

    def __str__(self):

        title = self.news_title if self.news_title else "News"

        return f"{title} - {self.prediction} ({self.confidence}%)"