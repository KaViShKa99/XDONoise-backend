from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from results import get_prediction
import traceback

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def clear_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)


@app.route('/upload-images', methods=['POST'])
def upload_images():
    try:

        clear_images(UPLOAD_FOLDER)
        clear_images("adv_imgs")
        clear_images("adv_xai_imgs")
        clear_images("original_xai_imgs")


        selected_images = request.json['selectedImages']

        for i, image_data in enumerate(selected_images):
            # Remove the prefix "data:image/jpeg;base64," from the URL
            image_url = image_data.split(',')[1]
            # Decode base64 data and save the image
            image_data_decoded = base64.b64decode(image_url)
            image_path = os.path.join(UPLOAD_FOLDER, f'image_{i}.jpg')
            with open(image_path, 'wb') as f:
                f.write(image_data_decoded)

        return jsonify({'message': 'Images saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)+" image not uploaded"}), 500

@app.route('/attack-parameters', methods=['POST'])
async def process_model():
    try:
        data = request.json
        selected_model = data.get('selectedModel')
        selected_parameters = data.get('selectedParameters')
        print(selected_model)

        adv_images = await get_prediction(selected_model,selected_parameters)
        
        return jsonify({'message': 'Attack data processed successfully','adv_images':adv_images})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
