import os

from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
import json
import subprocess
from django.http import JsonResponse
import pandas as pd
from .models import Picture
import math
from django.conf import settings
from urllib.parse import unquote
import base64

# Path to folder with input images -> created automatically when uploading images via website -> need to update path for local machine
input_image_directory = "C:\\Users\\danie\\Desktop\\Master\\Master SoSe 2023\\Machine Learning in Graphics, Vision and Language\\GithubTeamCode\\frontendPrototype\\prototypeSite\\media\\images\\"

# Path to run_inference.sh script on local machine -> find inference.sh file in your local files
script_path = "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/run_inference.sh"

# Path to csv file that saves all dot positions and image paths on local machine
# Make sure csv Excel file is closed when running inference -> otherwise error will occur
csv_path = "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode"


def gallery_view(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        for image in images:
            Picture.objects.create(image=image)
        return redirect('gallery')

    pictures = Picture.objects.all()
    return render(request, 'gallery.html', {'pictures': pictures})


def delete_images(request):
    if request.method == 'POST':
        selected_images = request.POST.getlist('selected_images')
        return render(request, 'delete_confirmation.html', {'selected_images': selected_images})
    return redirect('gallery')


def delete_images_confirm(request):
    if request.method == 'POST':
        selected_images = request.POST.get('selected_images').split(',')
        for image_id in selected_images:
            picture = Picture.objects.get(id=image_id)
            # Delete the image file from the image folder
            if os.path.exists(picture.image.path):
                os.remove(picture.image.path)
            # Delete the Picture object from the database
            picture.delete()
    return redirect('gallery')


def annotation_view(request, picture_id):
    picture = get_object_or_404(Picture, id=picture_id)
    pictures = Picture.objects.all()
    num_images = pictures.count()

    current_index = None
    prev_id = None
    next_id = None

    for index, pic in enumerate(pictures):
        if pic.id == picture_id:
            current_index = index
            break

    if current_index is not None:
        prev_index = (current_index - 1 + num_images) % num_images
        next_index = (current_index + 1) % num_images

        prev_id = pictures[prev_index].id
        next_id = pictures[next_index].id

    if request.method == 'POST' and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        x_coordinate = float(request.POST.get('x_coordinate'))
        y_coordinate = float(request.POST.get('y_coordinate'))
        picture.x_coordinate = x_coordinate
        picture.y_coordinate = y_coordinate
        picture.save()

        return JsonResponse({'success': True})

    # Collect the array list of data points with image ID, X, and Y coordinates
    data_points = [{'id': pic.id, 'x': pic.x_coordinate, 'y': pic.y_coordinate} for pic in pictures]

    context = {
        'picture': picture,
        'num_images': num_images,
        'current_image_index': current_index,
        'prev_picture_id': prev_id,
        'next_picture_id': next_id,
        'pictures': pictures,
        'data_points_json': json.dumps(data_points),  # Convert data_points to a JSON string
    }

    return render(request, 'annotation.html', context)


def extract_values(data):
    try:
        parsed_data = json.loads(data)
        resultExtractedArrayData = []

        for item in parsed_data:
            current_image_url = item.get("currentImageURL")
            x = math.ceil(float(item.get("x")))  # Round up to nearest integer
            y = math.ceil(float(item.get("y")))  # Round up to nearest integer

            url_parts = current_image_url.split("/")
            image_name = url_parts[-1]
            image_path = input_image_directory + image_name

            resultExtractedArrayData.append((image_path, x, y))

        return resultExtractedArrayData

    except json.JSONDecodeError:
        print("Invalid JSON data")


def submit_annotation(request):
    # Gather dot positions of annotated image
    if request.method == 'POST':
        dotPositions = request.POST.get('dotPositions')

        values = extract_values(dotPositions)

        # Create a pandas DataFrame
        df = pd.DataFrame(values, columns=["filepath", "x", "y"])

        # Save the DataFrame to an Excel file
        csv_filename = "annotation_image_data.csv"
        csv_full_path = os.path.join(csv_path, csv_filename)

        df.to_csv(csv_full_path, index=False, mode="w")

        print(f"CSV file '{csv_filename}' created/overwritten successfully at '{csv_full_path}'.")

        subprocess.call(script_path, shell=True)

        return redirect('prediction_results')

    # Return an error response if the request method is not POST
    return HttpResponse('Invalid request method.')


def base(request):
    return render(request, 'description.html')


def prediction_results(request):
    image_data = []
    prediction_image_folder = os.path.join(settings.PREDICTION_MEDIA_ROOT)
    all_image_files = [os.path.join(settings.PREDICTION_MEDIA_URL, f) for f in os.listdir(prediction_image_folder) if
                       f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file_counter in all_image_files:
        image_path = os.path.join(settings.PREDICTION_MEDIA_URL, image_file_counter)
        image_basename = os.path.basename(image_file_counter)
        image_data.append({'image_path': image_path, 'image_basename': image_basename})

    context = {
        'all_image_files': all_image_files,
        'image_location': prediction_image_folder,
        'image_data': image_data,
    }

    return render(request, 'prediction_results.html', context)


def clear_images_predictions(request):
    image_folder = os.path.join(settings.PREDICTION_MEDIA_ROOT)  # Use the second media root

    # Delete all image files in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_path = os.path.join(image_folder, filename)
            os.remove(file_path)

    # Redirect to another page after deleting images
    return redirect('gallery')
