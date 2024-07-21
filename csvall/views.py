from django.shortcuts import render
from django.shortcuts import render
from django.http import JsonResponse
from . import llmChain
import os


def search(request):
    if request.method == 'GET':
        return render(request, 'search.html')

    elif request.method == 'POST':
        prompt = request.POST.get('prompt', '')  # Get user input from form
        llm_name, best_answer, final_output_anthropic, final_output_vertex, final_output_openai\
              = llmChain.findBestAnswer(prompt)
        response_data = {
            "llm_name": llm_name,
            "best_answer": best_answer,
            "anthropic": final_output_anthropic,
            "vertex": final_output_vertex, 
            "openai": final_output_openai
        }
        return render(request, 'search.html', response_data)
