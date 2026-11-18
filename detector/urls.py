from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('check-news/', views.check_news, name='check_news'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('dashboard/clear-history/', views.clear_history, name='clear_history'),
    path('about/', views.about, name='about'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile, name='profile'),
]
