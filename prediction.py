import pandas as pd
import torch 
import os
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms.functional import normalize, resize,to_pil_image,to_tensor
from torchvision.io.image import read_image
from torchcam.utils import overlay_mask
from torchcam.methods import GradCAMpp
from torchvision import transforms
import time
from PIL import Image
import base64


from XDO_Methods import XDONoiseMethods
# from saveAdvImage import adv_img_save

vggface2_file_path = 'prediction_labels/labels-vggface2.csv'
vggface2_labels_df = pd.read_csv(vggface2_file_path)
vggface2_labels_array = vggface2_labels_df['Name'].values

casiaFace_file_path = 'prediction_labels/casiaFace_lables.csv'
casiaFace_labels_df = pd.read_csv(casiaFace_file_path)
casiaFace_labels_array = casiaFace_labels_df['Image'].values
casiaFace_image_paths_array = casiaFace_labels_df['Paths'].values

upload_dir = "uploads"
image_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]


# modified_tensor = None
model = None
class_label = None

def clear_images(directory):
      for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

def load_images_from_directory(directory):
    return [Image.open(os.path.join(directory, filename)) for filename in os.listdir(directory) if filename.endswith((".jpg", ".jpeg", ".png"))]


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def images_in_directory_to_base64(directory):
    return [base64.b64encode(open(os.path.join(directory, filename), "rb").read()).decode("utf-8") for filename in os.listdir(directory) if filename.endswith((".jpg", ".jpeg", ".png"))]


def adv_img_save(input_tensor, output_img_name):
   
    print("adv img 1")
    modified_tensor1 = torch.clamp(input_tensor, 0, 1)
    modified_image_pil = transforms.ToPILImage()(modified_tensor1)
    modified_image_pil.save(f"adv_imgs/{output_img_name}")

def xai_adv_image_save(model,last_layer,input_tensor,img_name):
    input_tensor = torch.clamp(input_tensor, 0, 1)

    with GradCAMpp(model,last_layer) as cam_extractor:
        output = model(input_tensor.unsqueeze(0))
        class_index = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_index, output)

    
    gradcam_img = overlay_mask(to_pil_image(input_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    gradcam_img.save(f"adv_xai_imgs/{img_name}")

def xai_image_save(model,last_layer,input_tensor,img_name):

    with GradCAMpp(model,last_layer) as cam_extractor:
        output = model(input_tensor.unsqueeze(0))
        class_index = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_index, output)

    
    gradcam_img = overlay_mask(to_pil_image(input_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    # modified_image_pil = transforms.ToPILImage()(input_tensor)
    gradcam_img.save(f"original_xai_imgs/{img_name}")

def calculate_magnitude_change(input_tensor, modified_tensor):
    squared_difference = torch.sum(torch.square(input_tensor - modified_tensor))
    magnitude_change = torch.sqrt(squared_difference)
    return magnitude_change.item()   

def prediction_prob_and_classLabel(model,selected_model,input_tensor):
    with torch.no_grad():

        output = model(input_tensor.unsqueeze(0))

        if selected_model == 'vggface2':
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class_index = torch.argmax(probs).item()
            max_prob, max_index = torch.max(probs, dim=1)
            accu = max_prob.item()*100
            class_label = vggface2_labels_array[predicted_class_index]

        elif selected_model == 'casiaWebFace':
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_prob, predicted_idx = torch.max(probs, 1)
            accu = predicted_prob.item()*100
            class_label = predicted_idx.item()

    return class_label , accu


def get_prediction(selected_model,selected_parameters):

    modified_tensor = None

    clear_images("adv_imgs")
    clear_images("adv_xai_imgs")
    clear_images("original_xai_imgs")

    print(selected_parameters)

    attack_name = selected_parameters["attackName"]
    attack_type = selected_parameters["type"]

    learning_rate = selected_parameters["lr"]
    threshold = selected_parameters["th"] # 0.1 - 0.9
    steps = selected_parameters["steps"]
    sample = selected_parameters["numSample"]

    pgd_epsilon = selected_parameters["eps"]
    pgd_iters = selected_parameters["iter"]


    if selected_model == 'vggface2':
        model = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
    elif selected_model == 'casiaWebFace':
        model = InceptionResnetV1(pretrained='casia-webface', classify=True).eval()

    xdo_noise_methods = XDONoiseMethods(model,'block8')

    adv_image_data = []
    adv_pred_label =[]
    adv_pred_prob = []
    mag_change = []
    ori_img_label = []


    for image_file in image_files:
        
        image_path = os.path.join(upload_dir, image_file)
        print(image_path)
        img = read_image(image_path)
        input_tensor = resize(img, (224, 224), antialias=False) / 255

        true_class_label , true_prediction_prob = prediction_prob_and_classLabel(model,selected_model,input_tensor)
        ori_img_label.append(true_class_label)

        input_image = input_tensor.clone().detach().requires_grad_(True)


        if attack_name == 'gradcampp' and attack_type == 'fgsm':
            modified_tensor = xdo_noise_methods.fgsm_attack_grad_pp(input_image, learning_rate, threshold)

        elif attack_name == 'gradcampp' and attack_type == 'pgd':
            modified_tensor = xdo_noise_methods.pgd_attack_grad_pp(input_image, pgd_epsilon, learning_rate, pgd_iters, threshold)

        elif attack_name == 'ig' and attack_type == 'fgsm':
            modified_tensor = xdo_noise_methods.fgsm_attack_IG(input_image, threshold, steps, learning_rate)

        elif attack_name == 'ig' and attack_type == 'pgd':
            modified_tensor = xdo_noise_methods.pgd_attack_IG(input_image, pgd_epsilon, learning_rate, pgd_iters, threshold, steps)

        elif attack_name == 'gradshap' and attack_type == 'fgsm':
            modified_tensor = xdo_noise_methods.fgsm_attack_gradient_shap(input_image, threshold, sample, learning_rate)

        elif attack_name == 'gradshap' and attack_type == 'pgd':
            modified_tensor = xdo_noise_methods.pgd_attack_gradient_shap(input_image, pgd_epsilon, learning_rate, pgd_iters, threshold, sample)

        adv_img_save(modified_tensor,f"{image_file}")

        magnitude_change = calculate_magnitude_change(input_tensor, modified_tensor)
        adversary_class_label , adversary_prediction_prob = prediction_prob_and_classLabel(model,selected_model,modified_tensor)

        adv_pred_label.append(adversary_class_label)
        adv_pred_prob.append(f"{adversary_prediction_prob:.2f}%")
        mag_change.append(f"{magnitude_change:.3f}")

        xai_image_save(model,'block8',input_tensor,f"{image_file}")

        xai_adv_image_save(model, 'block8', modified_tensor, f"{image_file}")

    ori_imgs = images_in_directory_to_base64("uploads") 
    adv_imgs = images_in_directory_to_base64("adv_imgs") 
    ori_xai_imgs = images_in_directory_to_base64("original_xai_imgs") 
    adv_xai_imgs = images_in_directory_to_base64("adv_xai_imgs")    

    image_data = {
        "ori_img":ori_imgs,
        "adv_img": adv_imgs,
        "true_label":ori_img_label,
        "adv_label": adv_pred_label,
        "adv_pred_prob": adv_pred_prob,
        "magnitude":mag_change,
        "ori_xai_img":ori_xai_imgs,
        "adv_xai_img":adv_xai_imgs
    }

    adv_image_data.append(image_data)

    
    return adv_image_data