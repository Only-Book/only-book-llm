# models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=255)
    price = models.CharField(max_length=255)
    description = models.TextField()
    publish_date = models.DateField()
    # category = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'book'
