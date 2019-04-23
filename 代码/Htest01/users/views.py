# users/views.py

from django.shortcuts import render
from .models import User
from .forms import AddForm
from . import test

# Create your views here.
def add(request):
    # 判断是否为 post 方法提交
    if request.method == "POST":
        af = AddForm(request.POST, request.FILES)
        # 判断表单值是否和法
        if af.is_valid():
            ctx = {}
            bf = AddForm()
            ctx['form'] = bf

            # name = af.cleaned_data['name']
            name = None
            headimg = af.cleaned_data['headimg']

            answer = test.predict(headimg)

            user = User(name=name, headimg=headimg)
            user.save()
            # answer = test.predict(str(headimg))

            ctx['user'] = user
            ctx['answer'] = answer
            return render(request, 'users/index.html', context={"ctx": ctx})

    else:
        ctx = {}
        af = AddForm()
        ctx['form'] = af
        ctx['headimg'] = None
        ctx['result'] = None
        ctx['accuracy'] = None
        return render(request, 'users/index.html', context={"ctx": ctx})