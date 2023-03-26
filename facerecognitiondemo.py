from os import listdir
import face_recognition
import cv2
import time


class FaceRecognitionDemo:
    known_face_encodings = []
    known_face_names = []

    def load_samples(self, path):
        for file in listdir(path):
            if file.endswith('jpg') or file.endswith('png'):
                sample_pic = face_recognition.load_image_file(path + '/' + file)
                self.known_face_encodings.append(face_recognition.face_encodings(sample_pic)[0])
                self.known_face_names.append(file.rsplit('.')[0])

    def recognize(self, target_image):
        image_encoding = face_recognition.face_encodings(target_image)
        if len(image_encoding) > 0:
            results = face_recognition.compare_faces(self.known_face_encodings, image_encoding[0])
            name = 'Unknown'
            if not True in results:
                print('Unknown face')
            else:
                name = self.known_face_names[results.index(True)]
            return name, face_recognition.face_locations(target_image)[0]
        return '', [0, 0, 0, 0]


if __name__ == "__main__":
    # start camera
    cap = cv2.VideoCapture(0)
    face_rec = FaceRecognitionDemo()
    face_rec.load_samples('face_samples')
    rec_times = 0
    y1, x1, y2, x2 = 0, 0, 0, 0
    face_name = ''
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            if rec_times % 5 == 0:
                face_name, face_shape = face_rec.recognize(frame)
                y1, x1, y2, x2 = face_shape[0], face_shape[1], face_shape[2], face_shape[3]
                rec_times = 0
            rec_times = rec_times + 1
            cv2.putText(frame, face_name, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.imshow('video', frame)
            time.sleep(0.01)
        else:
            break
        if cv2.waitKey(10) == ord('q'):
            break
