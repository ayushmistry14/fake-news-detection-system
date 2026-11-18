from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import LoginForm, NewsForm, RegisterForm
from .models import PredictionHistory
from .utils import predict_news


def home(request):
    return render(request, 'detector/home.html')


def check_news(request):
    form = NewsForm(request.POST or None)
    result = None
    confidence = None
    summary = None
    news_title = None
    
    if request.method == 'POST' and form.is_valid():
        news_title = form.cleaned_data.get('news_title', '')
        news_text = form.cleaned_data['news']
        result, confidence = predict_news(news_title,news_text)
        if result == 'Real News':
            summary = 'The given news content is most likely to be real based on the available analysis.'
        elif result == 'Fake News':
            summary = 'The given news content is most likely to be fake based on the available analysis.'
        else:
            summary = 'The model is not confident enough to classify this news as real or fake.'
        PredictionHistory.objects.create(
            user=request.user if request.user.is_authenticated else None,
            news_title=news_title,
            news_text=news_text,
            prediction=result,
            confidence=confidence,
        )

    context = {
        'form': form,
        'result': result,
        'confidence': confidence,
        'summary': summary,
        'news_title': news_title,
    }
    return render(request, 'detector/check_news.html', context)


@login_required(login_url='detector:login')
def dashboard(request):
    history = PredictionHistory.objects.filter(user=request.user).order_by('-created_at')[:10]
    total = PredictionHistory.objects.filter(user=request.user).count()
    fake_count = PredictionHistory.objects.filter(user=request.user, prediction='Fake News').count()
    real_count = PredictionHistory.objects.filter(user=request.user, prediction='Real News').count()

    context = {
        'history': history,
        'total': total,
        'fake_count': fake_count,
        'real_count': real_count,
    }
    return render(request, 'detector/dashboard.html', context)


@login_required(login_url='detector:login')
def clear_history(request):
    if request.method == 'POST':
        PredictionHistory.objects.filter(user=request.user).delete()
        messages.success(request, 'Prediction history cleared successfully.')
    return redirect('detector:dashboard')


def about(request):
    return render(request, 'detector/about.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('detector:dashboard')

    form = LoginForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('detector:dashboard')
        messages.error(request, 'Invalid username or password.')

    return render(request, 'detector/login.html', {'form': form})


def register_view(request):
    if request.user.is_authenticated:
        return redirect('detector:dashboard')

    form = RegisterForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            messages.success(request, 'Registration successful. Please log in.')
            return redirect('detector:login')
        messages.error(request, 'Please correct the errors below.')

    return render(request, 'detector/register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('detector:home')


@login_required(login_url='detector:login')
def profile(request):
    history = PredictionHistory.objects.filter(user=request.user).order_by('-created_at')[:5]
    total = PredictionHistory.objects.filter(user=request.user).count()
    fake_count = PredictionHistory.objects.filter(user=request.user, prediction='Fake News').count()
    real_count = PredictionHistory.objects.filter(user=request.user, prediction='Real News').count()

    context = {
        'history': history,
        'total': total,
        'fake_count': fake_count,
        'real_count': real_count,
    }
    return render(request, 'detector/profile.html', context)
