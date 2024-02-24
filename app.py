from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import base64
from easyocr import Reader
import pandas as pd
import avro

app = Flask(__name__)

# Create a folder named "generated_pic" if it doesn't exist
generated_folder = os.path.join(os.getcwd(), "generated_pic")
os.makedirs(generated_folder, exist_ok=True)

# Load the dataset into memory
dataset_path = "Dataset_Final_0.csv"
dataset = pd.read_csv('Dataset_Final_0.csv')
dataset['Banglish'] = dataset['Banglish'].str.lower()
dataset['Bangla'] = dataset['Bangla'].str.lower()

# Step 2: Data Cleaning
dataset['Banglish_words'] = dataset['Banglish'].apply(lambda x: len(x.split()))
dataset['Bangla_words'] = dataset['Bangla'].apply(lambda x: len(x.split()))
dataset = dataset[dataset['Banglish_words'] == dataset['Bangla_words']]


# Step 3: Create Word Mapping Dictionary
word_mapping = {}
for index, row in dataset.iterrows():
    banglish_words = row['Banglish'].split()
    bangla_words = row['Bangla'].split()
    for banglish, bangla in zip(banglish_words, bangla_words):
        word_mapping[banglish] = bangla

# Step 4-6: Input Processing and Result

# def generate_bangla_sentence(input_banglish_sentence):
#     input_banglish_words = input_banglish_sentence.lower().split()
#     output_bangla_sentence = []
#     for word in input_banglish_words:
#         if word in word_mapping:
#             output_bangla_sentence.append(word_mapping[word])
#         else:
#             # Use avro.parse for phonetic transliteration
#             bangla_word = avro.parse(word)
#             output_bangla_sentence.append(bangla_word)
#     return ' '.join(output_bangla_sentence)

def generate_bangla_sentence(input_banglish_sentence):
    input_banglish_words = input_banglish_sentence.lower().split()
    output_bangla_sentence = []
    for word in input_banglish_words:
        if word in word_mapping:
            output_bangla_sentence.append(word_mapping[word])
        else:
            # Use avro.parse for phonetic transliteration
            bangla_word = avro.parse(word)
            output_bangla_sentence.append(bangla_word)
    return ' '.join(output_bangla_sentence)



# def map_words(english_text):
#     """
#     Map English words to Bengali words based on the dataset.
#     """
#     bengali_words = []
#     english_words = english_text.split()
#     for word in english_words:
#         # Check if the English word exists in the dataset
#         if word in dataset['Banglish'].values:
#             # If yes, retrieve the corresponding Bengali word
#             bengali_word = dataset.loc[dataset['Banglish'] == word, 'Bangla'].values[0]
#             bengali_words.append(bengali_word)
#         else:
#             # If not, handle it differently (e.g., return the English word itself)
#             bengali_words.append(word)
#     return ' '.join(bengali_words)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generated_pic/<filename>')
def generated_pic(filename):
    return send_from_directory(generated_folder, filename)


# @app.route('/save_drawing', methods=['POST'])
# def save_drawing():
#     data = request.json['data']
#     filename = 'drawing.png'  # Use a fixed filename for overwriting the previous image
#     filepath = os.path.join(generated_folder, filename)

#     with open(filepath, 'wb') as f:
#         f.write(base64.b64decode(data.split(',')[1]))

#     return jsonify({'status': 'success', 'filename': filename})

@app.route('/save_drawing', methods=['POST'])
def save_drawing():
    data = request.json.get('data', None)
    if data:
        # Check if the data URI scheme is present
        if 'data:image/png;base64,' in data:
            # Split the data URI scheme to extract the base64 encoded image data
            base64_data = data.split(',')[1]
            filename = 'drawing.png'  # Use a fixed filename for overwriting the previous image
            filepath = os.path.join(generated_folder, filename)
            
            # Decode the base64 encoded image data and write it to the file
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(base64_data))
            return jsonify({'status': 'success', 'filename': filename})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data format'})
    else:
        return jsonify({'status': 'error', 'message': 'No data received'})

# @app.route('/extract_text', methods=['POST'])
# def extract_text():
#     filename = 'drawing.png'
#     filepath = os.path.join(generated_folder, filename)

#     # Use EasyOCR to extract text from the image
#     reader = Reader(['en'])
#     results = reader.readtext(filepath)

#     # Extracted text
#     extracted_text = ' '.join([result[1] for result in results])
#     print("Extracted English Text:", extracted_text)  # Add this line for debugging

#     # Bangla text
#     banglai_text = generate_bangla_sentence(extracted_text)  # Use the function to map words
#     print("Translated Bangla Text:", banglai_text)  # Add this line for debugging

#     return jsonify({
#         'status': 'success', 
#         'english_text': extracted_text,  
#         'banglai_text': banglai_text     
#     })
    
@app.route('/extract_text', methods=['POST'])
def extract_text():
    data = request.json.get('data', None)
    if data:
        # Check if the data URI scheme is present
        if 'data:image/png;base64,' in data:
            # Split the data URI scheme to extract the base64 encoded image data
            base64_data = data.split(',')[1]
            
            # Decode the base64 encoded image data
            image_data = base64.b64decode(base64_data)
            
            # Use EasyOCR to extract text from the image
            reader = Reader(['en'])
            results = reader.readtext(image_data)

            # Extracted text
            extracted_text = ' '.join([result[1] for result in results])
            print("Extracted English Text:", extracted_text)  # Add this line for debugging

            # Translate English text to Bangla using avro.parse
            banglai_text = generate_bangla_sentence(extracted_text)
            print("Translated Bangla Text:", banglai_text)  # Add this line for debugging

            return jsonify({
                'status': 'success', 
                'english_text': extracted_text,  
                'banglai_text': banglai_text     
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data format'})
    else:
        return jsonify({'status': 'error', 'message': 'No data received'})

if __name__ == '__main__':
    app.run(debug=True)