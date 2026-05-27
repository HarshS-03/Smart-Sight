from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    code = models.CharField(max_length=100, unique=True, null=True, blank=True)
    
    def __str__(self):
        return self.username

# Surveillance Models
class Person(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

def person_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/dataset/<person_name>/<filename>
    return f'dataset/{instance.person.name}/{filename}'

class PersonImage(models.Model):
    person = models.ForeignKey(Person, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=person_directory_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.person.name}"

class RecognitionLog(models.Model):
    STATUS_CHOICES = [
        ('KNOWN', 'Known'),
        ('UNKNOWN', 'Unknown'),
    ]
    person_name = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    image_path = models.CharField(max_length=255, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.person_name if self.person_name else 'Unknown'} - {self.status} at {self.timestamp}"
