import cv2

def generate_dataset(img, id, img_id):
    cv2.imwrite("pic/user."+str(id)+"_"+str(img_id)+".jpg", img)



def draw_bound(img, classifier, scaleFactor, minNeighbour, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbour)
    coords = []
    for [x, y, w, h] in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, _ =clf.predict(gray_img[y:y+h, x:x+w])
        if id == 1:
            cv2.putText(img, "Saniya", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img

def recognize(img, clf, faceCascade):
    color = {"blue":(255, 0, 0), "red":(0, 0, 255), "green":(0, 255, 0), "white":(255, 255, 255)}
    coords= draw_bound(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img

def detect(img, faceCascade, eyeCascade, img_id):
    color = {"blue":(255, 0, 0), "red":(0, 0, 255), "green":(0, 255, 0), "white":(255, 255, 255)}
    coords, img = draw_bound(img, faceCascade, 1.1, 10, color['blue'], "Face ")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)
        # coords = draw_bound(roi_img, eyeCascade, 1.1, 8, color['red'], "Eyes")
        # coords = draw_bound(roi_img, noseCascade, 1.1, 5, color['green'], "Nose")
        # coords = draw_bound(roi_img, mouthCascade, 1.1, 20, color['red'], "Mouth")
    return img


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
# noseCascade = cv2.CascadeClassifier("nose.xml")
# mouthCascade = cv2.CascadeClassifier("Mouth.xml")

clf= cv2.face.LBPHFaceRecognizer_create()
clf.read("classifierP.yml")
video_capture= cv2.VideoCapture(0)

img_id = 0

while True:
    _, img= video_capture.read()
    # img = detect(img, faceCascade, eyeCascade,img_id)
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face detetcion", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # fileName= "MyImage.jpg"
        # cv2.imwrite(fileName, img)
        # print("image Capture Successfully")
        break
video_capture.release()
cv2.destroyAllWindows()


