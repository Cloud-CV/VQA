from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from channels import Group

from .sender import vqa_task
from .utils import log_to_terminal, prepro_question

import demo.constants as constants
import uuid
import os
import random
import traceback
import urllib2


def vqa(request, template_name="vqa/vqa.html"):
    socketid = uuid.uuid4()
    if request.method == "POST":
        # get the parameters from client side
        try:
            socketid = request.POST.get('socketid')
            vqa_model = request.POST.get('vqa_model', 'pythia')
            input_question = request.POST.get('question', '')
            input_question = input_question.lower().replace("?", "")
            input_question = "{0} {1}".format(input_question, "?")

            img_path = request.POST.get('img_path')
            img_path = urllib2.unquote(img_path)

            print request.POST
            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the VQA wrapper
            log_to_terminal(socketid, {"terminal": "Starting Visual Question Answering job..."})
            response = vqa_task(str(abs_image_path), str(input_question), str(vqa_model), socketid)
        except Exception, err:
            log_to_terminal(socketid, {"terminal": traceback.print_exc()})

    demo_images = get_demo_images(constants.COCO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images, 'socketid': socketid})


def file_upload(request):
    if request.method == "POST":
        image = request.FILES['file']
        socketid = request.POST.get('socketid')
        dir_type = constants.VQA_CONFIG['image_dir']

        random_uuid = uuid.uuid1()
        # handle image upload
        output_dir = os.path.join(dir_type, str(random_uuid))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_path = os.path.join(output_dir, str(image))
        handle_uploaded_file(image, img_path)

        img_url = img_path.replace(settings.BASE_DIR, "")

        return JsonResponse({"file_path": img_path.replace(settings.BASE_DIR, '')})    


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def get_demo_images(demo_images_path):
    try:
        image_count = 0
        demo_images = []
        while(image_count<6):
            random_image = random.choice(os.listdir(demo_images_path))
            if "COCO_" in random_image:
                demo_images.append(random_image)
                image_count += 1

        demo_images = [os.path.join(constants.COCO_IMAGES_URL, x) for x in demo_images]
        print demo_images
    except Exception as e:
        print e
        images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg',]
        demo_images = [os.path.join(settings.STATIC_URL, 'images', x) for x in images]

    return demo_images
