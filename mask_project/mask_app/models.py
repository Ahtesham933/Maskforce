from django.db import models

# Create your models here.

class DetectionLog(models.Model):
    mask_detected = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    image_path = models.ImageField(upload_to='captures/', blank=True, null=True)
    confidence = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-timestamp']
        db_table = 'detection_log'

    def __str__(self):
        status = "Mask" if self.mask_detected else "No Mask"
        return f"{status} | {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {self.confidence:.2f}"