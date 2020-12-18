import numpy as np
import cv2
import face_recognition
import time

flag = None

'''
7amada_image = face_recognition.load_image_file(Path)
7amada_Enco = face_recognition.face_encodings(7amada_image)

joeM_image = face_recognition.load_image_file(Path)
joeM_Enco = face_recognition.face_encodings(joeM_image)

#Array to hold face encodings
known_face_encodings = [
    7amada_Enco,
    joeM_Enco
    ]
   #Array to hold names 
    known_face_names = [
    "7amada",
    "joeM"
]





'''

My_image = face_recognition.load_image_file('Me1.jpeg')
My_Enco = face_recognition.face_encodings(My_image)
My_Name = ["Operator"]
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FInd encodings and locations in frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(My_Enco, face_encoding)

        name = "Unknown"
        #Matching encodings with pre trained set
        face_distances = face_recognition.face_distance(My_Enco, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = "Omar"

        # Bounding detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)

        # Assign a name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name == 'Omar':

            flag = 1
            time.sleep(1)
            print('Welcome Home')
        elif 'Unknown' == name:
            flag = 0
            print('Use keypad to enter')
        elif name == None:
            flag = None

    cv2.imshow('Video', frame)
    print(flag)
    #Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release n Flush!
video_capture.release()
cv2.destroyAllWindows()
