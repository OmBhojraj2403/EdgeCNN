import cv2 
import tensorflow as tf
import numpy as np
import datetime
import sys
import os

def time_in_seconds(x):
    start_time = x[0][0]*60 + x[0][1]
    end_time = x[1][0]*60 + x[1][1]
    return [start_time, end_time]

def predict_tflite(interpreter, frame, img_size=44):
    # Getting the input and output tensor details from the model graph.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.array(img[:,450:], dtype=np.float32)

    # Tensorflow model preprocess_inputs if required
    img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
    processed_frame = np.expand_dims(img, 0) # (1,224,224,3)
    placeholder_input = np.array(processed_frame, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], placeholder_input)

    # Getting output from the tflite model
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    label = np.argmax(output, axis=1)
    score = output[0][label]
    return label[0], score[0]

def predict_tf(interpreter, frame, img_size=44):
    # Getting the input and output tensor details from the model graph.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = np.array(img[:,450:], dtype=np.float32)
    # Tensorflow model preprocess_inputs if required
    img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
    processed_frame = np.expand_dims(img, 0) # (1,224,224,3)
    output = interpreter.predict(processed_frame, verbose=0)
    label = np.argmax(output, axis=1)
    score = output[0][label]
    return label[0], score[0]


model_name = ""
model_path = f"./weights/{model_name}"
# tflite_model_name = ""
# model_path = 'tflite_weights/{tflite_model_name}'

type = None
if model_path.split('.')[-1]=="tflite":
    type = "tflite"
else:
    type = "tf"

# model = None
if type=="tf":
    model = tf.keras.models.load_model(model_path)
else:
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model = interpreter

vid_path = "" # Enter Video path if any
if vid_path == "":
    vid = 0
else: 
    vid = vid_path
vid = cv2.VideoCapture(vid)

# Format for timings - [[start_minute, start_second],[end_minute, end_second]]
inference_timings = [[0,0],[1,0]]
inference_duration = time_in_seconds(inference_timings)

saved_vid_name = "vid_inf_res" # Name of saved video  
out_path = f'video_inference/{saved_vid_name}.mp4'
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, codec, fps, (width, height))

classes = [] # Fill the class names as strings

font = cv2.FONT_HERSHEY_SIMPLEX # font
org = (100, 100) # org
fontScale = 2 # fontScale 
color = (255, 255, 255) # Blue color in BGR
thickness = 3 # Line thickness of 2 px

print("Saving inference video")

frame_count = 0
entered_video = 0


while True:

    ret, frame1 = vid.read()
    if not ret:
        if entered_video==0:
            print("Video not found")
            break
        else:
            print("\nVideo inference done")
            break

    frame_count+=1
    time_elapsed_sec = frame_count/fps
    if time_elapsed_sec>inference_duration[1]:
        print("\nVideo inference done")
        break

    minutes = f'{int(time_elapsed_sec//60)}'
    sec = int(time_elapsed_sec)%60
    seconds = f'{sec}' if sec>9 else f'0{sec}'
    print(f'\rVideo time elapsed: {minutes}:{seconds} ',  end="")

    entered_video=1
    if (inference_duration[0] < time_elapsed_sec < inference_duration[1]):     
        # Running inference on the frame
        img = frame.copy()
        if type == "tflite":
            label, score = predict_tflite(model, img) #TFLite model Prediction
        else:
            label, score = predict_tf(model, img) #TF model prediction
        prob = score*100
        prediction = classes[int(label)]
        to_print = str(prediction) + "," + str(round(prob,2))
    
        frame = cv2.putText(frame, to_print, org, font, fontScale, color, thickness, cv2.LINE_AA) #Writing result on the frame
        out.write(frame) # Save frame to output video 
    
    cv2.imshow("frame", frame) # Dispaying the processed frames
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


vid.release()
out.release()
cv2.destroyAllWindows()
print("Inference saved")
print('\nEnd time: ',datetime.datetime.now())