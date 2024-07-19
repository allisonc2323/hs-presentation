from django.shortcuts import render
from django.shortcuts import render
from django.http import JsonResponse
from . import llmChain
import os


def prompt_form(request):
    if request.method == 'GET':
        return render(request, 'prompt_form.html')

    elif request.method == 'POST':
        prompt = request.POST.get('prompt', '')  # Get user input from form
        llm_name, best_answer = llmChain.findBestAnswer(prompt)
        response_data = {
            'llm_name': llm_name,
            'best_answer': best_answer,
        }
        return JsonResponse(response_data)
