## Descripciones de películas

En esta etapa del proyecto se utilizará la API de OpenAI para generar descripciones de algunas películas que se agregarán a la base de datos.
Para esto, el archivo ``movie_titles.json`` que se encuentra en la raíz del proyecto, tiene una lista de películas a las que se le agregará información.

El archivo [movie_descriptions.py](movie_descriptions.py) muestra los pasos para utilizar la API de OpenAI para obtener la descripción de las películas en la lista.
En este archivo se pueden ver dos funcionalidades principales:

1. Definición de una función auxiliar (__get_completion__) para comunicarse con la API
2. Creación de un __prompt__ que nos ayuda a pedir de forma correcta la descripción de las películas

Ejecute el script, para esto debe dirigirse en la consola a la ubicación del archivo y escribir:

````bash
python movie_descriptions.py
````

Cuando ejecute el script, deberá ver en la consola algo de la siguiente forma:

![Fork 1](imgs/md1.png)


Puede ver que se imprime el nombre de la película, el prompt completo y la descripción obtenida.

Al ejecutar todo el script (incluyendo las lineas comentadas) se genera el archivo ``movie_descriptions.json``
que se va a utilizar para alimentar la base de datos de películas. En este caso, por tiempo, no se va a ejecutar el script completo y el archivo resultante se puede consultar en [movie_descriptions.json](movie_descriptions.json)


Ahora se utilizará la información en el archivo [movie_descriptions.json](movie_descriptions.json) para agregar items a la base de datos. Para esto nos vamos a dirigir a la carpeta ``DjangoProjectBase``. Asumiendo que la consola está en el directorio raíz del proyecto ``ProyectoPeliculas2023-2_Taller3``, escriba lo siguiente:

````shell
cd DjangoProjectBase
````
Si ejecuta el servidor se dará cuenta que este es el proyecto que se creó en el workshop 2

````shell
python manage.py runserver
````
![Fork 1](imgs/md2.png)

__Nota:__ Antes de continuar es necesario __borrar__ la base de datos existente, hacer las migraciones, y crear de nuevo las credenciales de super-usuario.

````shell
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
````

Ahora dentro de la carpeta de la app movie debe crear una carpeta management y dentro de esta una carpeta commands. Después, debe crear el archivo add_descriptions_db.py.

![Fork 1](imgs/md3.png)

Este archivo se utilizará para pasar la información del archivo ``movie_descriptions.json`` a la base de datos de películas de la aplicación de Django. El contenido de este archivo se encuentra en [add_descriptions_db.py](aux_files/add_descriptions_db.py)

Para evitar un error dado que las películas no tienen imágenes, debe ir al archivo ``movie/models.py`` y modificar la línea 8 de la siguiente forma:

````python
image = models.ImageField(upload_to='movie/images/', default = 'movie/images/default.jpg')
````
![Fork 1](imgs/md6.png)

Dado que se modificó ``models.py`` se deben hacer las migraciones.

````shell
python manage.py makemigrations
python manage.py migrate
````

Además, en la carpeta ``media/movie/images`` debe agregar la figura [default.jpg](aux_files/default.jpg) (puede ser cualquier figura).

Una vez haya terminado estos pasos y copiado el contenido del archivo [add_descriptions_db.py](aux_files/add_descriptions_db.py) en ``movie/management/commands/add_descriptions.py``, en la consola va a ejecutar el siguiente comando:

````shell
python manage.py add_descriptions_db
````
Cuando termine de ejecutarse, debe ver un mensaje como el siguiente:

![Fork 1](imgs/md4.png)

Puede ejecutar el servidor y verá algo de la siguiente forma:

![Fork 1](imgs/md7.png)

Además puede ir a la página de administrador 127.0.0.1:8000/admin/ y cuando ingrese con las credenciales podrá observar que las películas quedaron correctamente almacenadas en la base de datos. Además, puede ingresar a alguna de ellas y ver la descripción

![Fork 1](imgs/md8.png)




