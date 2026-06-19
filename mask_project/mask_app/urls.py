from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/detect/', views.detect_api, name='detect_api'),
    path('api/logs/', views.logs_api, name='logs_api'),
    path('api/stats/', views.stats_api, name='stats_api'),
    path('api/delete/', views.delete_logs_api, name='delete_logs_api'),
    path('api/export/', views.export_logs_api, name='export_logs_api'),
]