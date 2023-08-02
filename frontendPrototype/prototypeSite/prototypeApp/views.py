import os

from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
import json
import subprocess
from django.http import JsonResponse

from .models import Picture


# ToDo: Add specific annotation view
# ToDo: Test script functionality

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
    print("Going to annotation view")
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

    print("arrived in annotation view")
    return render(request, 'annotation.html', context)


def submit_annotation(request):
    print("in anno post method")
    if request.method == 'POST':
        data_points_json = request.POST.get('data_points_json')
        data_points = json.loads(data_points_json)
        dot_positions = request.POST.get('dotPositions')

        print('Dot Positions:', dot_positions)
        print("Data Points: ", data_points)
        subprocess.call(
            "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/run_inference.sh",
            shell=True)

        # ToDo: save results as array -> mask + image link
        # Loop through images to get images needed to be rendered

        context = {
            'data_points': data_points,
        }

        # return HttpResponse('Annotation submitted successfully.')
        print("going to results")
        return redirect('result', context)

    # Return an error response if the request method is not POST
    return HttpResponse('Invalid request method.')


def base(request):
    return render(request, 'description.html')


def result(request):
    print("arrived in results")
    dot_positions = request.POST.get('dotPositions')
    print("result dot positions: ", dot_positions)

    pictures = Picture.objects.all()
    num_images = pictures.count()
    allDotPositions = dot_positions
    imageLinks = []

    current_index = None
    prev_id = None
    next_id = None

    if current_index is not None:
        prev_index = (current_index - 1 + num_images) % num_images
        next_index = (current_index + 1) % num_images

        prev_id = pictures[prev_index].id
        next_id = pictures[next_index].id

        return JsonResponse({'success': True})

    context = {
        'num_images': num_images,
        'current_image_index': current_index,
        'prev_picture_id': prev_id,
        'next_picture_id': next_id,
        'pictures': pictures,
    }

    return render(request, 'result.html')
