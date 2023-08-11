"""
URL configuration for prototypeSite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

# TODO: fix import error
from prototypeApp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('gallery/', views.gallery_view, name='gallery'),
    path('annotation/<int:picture_id>/', views.annotation_view, name='annotation'),
    path('submit', views.submit_annotation, name='submit_annotation'),
    path('', views.base, name='base'),
    path('delete/', views.delete_images, name='delete_images'),
    path('delete/confirm/', views.delete_images_confirm, name='delete_images_confirm'),
    path('results/', views.prediction_results, name='prediction_results'),
    path('clear_images/', views.clear_images_predictions, name='clear_images_predictions'),
    path('image_detail/<str:image_name>/', views.image_detail, name='image_detail'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.PREDICTION_MEDIA_URL, document_root=settings.PREDICTION_MEDIA_ROOT)
