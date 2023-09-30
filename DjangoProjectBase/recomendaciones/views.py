from django.shortcuts import render
from movie.models import Movie
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
from dotenv import load_dotenv, find_dotenv

def recomendaciones(request):
    searchTerm = request.GET.get('searchMovie')
    if searchTerm: 
        _ = load_dotenv('../openAI.env')
        openai.api_key  = os.environ['openAI_api_key']

        with open('../movie_descriptions_embeddings.json', 'r') as file:
            file_content = file.read()
            movies = json.loads(file_content)

        #Esta función devuelve una representación numérica (embedding) de un texto, en este caso
        #la descripción de las películas
        emb = get_embedding(movies[1]['description'],engine='text-embedding-ada-002')

        #Vamos a crear una nueva llave con el embedding de la descripción de cada película en el archivo .json
        emb = get_embedding(searchTerm,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(movies)):
            sim.append(cosine_similarity(emb,movies[i]['embedding']))
        sim = np.array(sim)
        idx = np.argmax(sim)

        movies = Movie.objects.filter(title__icontains=movies[idx]['title']) 
    else: 
        movies = Movie.objects.all()
    return render(request, 'recomendaciones.html', {'searchTerm':searchTerm, 'movies': movies})


        # Create your views here.
