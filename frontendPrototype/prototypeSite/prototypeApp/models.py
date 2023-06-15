from django.db import models


class Meta:
    app_label = 'prototypeApp'


class Picture(models.Model):
    image = models.ImageField(upload_to='images/')
    annotation = models.TextField()


