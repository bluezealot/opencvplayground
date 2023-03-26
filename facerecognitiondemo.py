from os import listdir
import face_recognition
import cv2
import numpy as np


class FaceRecognitionDemo:
    known_face_encodings = []
    known_face_names = []
    process_frame = True
    face_locations = []
    face_names = []

    def load_samples(self, path):
        for file in listdir(path):
            if file.endswith('jpg') or file.endswith('png'):
                sample_pic = face_recognition.load_image_file(path + '/' + file)
                self.known_face_encodings.append(face_recognition.face_encodings(sample_pic)[0])
                self.known_face_names.append(file.rsplit('.')[0])

    def recognize(self, target_image):
        locations = face_recognition.face_locations(target_image)
        image_encodings = face_recognition.face_encodings(target_image, locations)
        names = []
        for image_encoding in image_encodings:
            results = face_recognition.compare_faces(self.known_face_encodings, image_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, image_encoding)
            best_match = np.argmin(face_distances)
            name = 'Unknown'
            if results[best_match]:
                name = self.known_face_names[best_match]
            names.append(name)
        return names, locations

    def process_frame_recognize(self, target_frame):
        if self.process_frame:
            small_frame = cv2.resize(target_frame, (0, 0), fx=0.25, fy=0.25)
            small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            self.face_names, self.face_locations = self.recognize(small_rgb)
        self.process_frame = not self.process_frame
        for face_name, (y1, x1, y2, x2) in zip(self.face_names, self.face_locations):
            y1 *= 4
            x1 *= 4
            y2 *= 4
            x2 *= 4
            cv2.putText(target_frame, face_name, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.rectangle(target_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        return target_frame


if __name__ == "__main__":
    face_rec = FaceRecognitionDemo()
    face_rec.load_samples('face_samples')
    # start camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            processed_frame = face_rec.process_frame_recognize(frame)
            cv2.imshow('recognized', processed_frame)
        else:
            break
        if cv2.waitKey(10) == ord('q'):
            break
