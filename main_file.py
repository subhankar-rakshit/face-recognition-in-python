# Import face recognition
import face_recognition as fr
import numpy as np
import os
import cv2

def encode_faces():
    encoded_data = {}

    for dirpath, dnames, fnames in os.walk("./Images"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("Images/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded_data[f.split(".")[0]] = encoding

    # return encoded data of images
    return encoded_data

# Function for face detection & recognition
def detect_faces():
    faces = encode_faces()
    encoded_faces = list(faces.values())
    faces_name = list(faces.keys())

    video_frame = True
    # Capturing video through the WebCam
    # Real Time Video Streams
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()

        if video_frame:
            face_locations = fr.face_locations(frame)
            unknown_face_encodings = fr.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in unknown_face_encodings:
                # Comapring faces
                matches = fr.compare_faces(encoded_faces, face_encoding)
                name = "Unknown"

                face_distances = fr.face_distance(encoded_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = faces_name[best_match_index]

                face_names.append(name)

        video_frame = not video_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a rectangular box around the face
            cv2.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (0, 255, 0), 2)
            # Draw a Label for showing the name of the person
            cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # Showing the name of the detected person through the WebCam
            cv2.putText(frame, name, (left -20, bottom + 15), font, 0.85, (255, 255, 255), 2)
            
        cv2.imshow('Video', frame)
        code = cv2.waitKey(1)
        # Press 'q' for close the video frame
        if code == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
