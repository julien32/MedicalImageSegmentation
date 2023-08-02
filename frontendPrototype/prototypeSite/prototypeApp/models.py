from django.db import models


class Meta:
    app_label = 'prototypeApp'


class Picture(models.Model):
    image = models.ImageField(upload_to='images/')
    x_coordinate = models.FloatField(null=True, blank=True)
    y_coordinate = models.FloatField(null=True, blank=True)
    annotation = models.TextField()

    def __str__(self):
        return f"Picture {self.id}"
