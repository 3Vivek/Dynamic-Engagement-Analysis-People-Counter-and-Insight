from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*

#by default 0 will open live cam
# cap=cv2.VideoCapture(0)
# cap.set(3,1280) #width id is  3
# cap.set(4,720) #height id is 4
# cap.set(10,100) # brightness id is 10

cap = cv2.VideoCapture("./Videos/cars.mp4")  # For Video

model=YOLO('./YOLO_Weight/yolov8n.pt')


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv2.imread("Project_1_car_counter/mask.png")
#Tracking 
#max_age=limit of number of frame we can still recognize the car when it come back 

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

line_limit=[400,297,673,297] 
totalCount=[]
while True:
    istrue,frame=cap.read()
    #do bitwise (and) only show those region that are in mask
    imgRegion=cv2.bitwise_and(frame,mask) 
    imgGraphics=cv2.imread("Project_1_car_counter/graphics.png",cv2.IMREAD_UNCHANGED)
    
    frame=cvzone.overlayPNG(frame,imgGraphics,(0,0))
    
    res=model(imgRegion,stream=True)

    detection=np.empty((0,5))

    for r in res:
        boxes=r.boxes
        for box in boxes:
            #Bounding box

            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)

            # x1,y1,w,h=box.xywh[0]
            w,h=x2-x1,y2-y1
            # cvzone.cornerRect(frame,(x1,y1,w,h),t=5)

            #Confidence

            conf=math.ceil((box.conf[0]*100) )/100
            print(conf)
            # cvzone.putTextRect(frame,f'{conf}',(x1,y1-20))
            #aisa kro ki text bahar na jae
            # cvzone.putTextRect(frame,f'{conf}',(max(0,x1),max(35,y1)))

            #class Name
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass=="car" or currentClass=="truck" or currentClass=="bus"\
                    or currentClass=="motorbike" and conf>0.3:
                # cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6,thickness=1,offset=3)
                # cvzone.cornerRect(frame,(x1,y1,w,h),l=4,rt=5)

                currentArray=np.array([x1,y1,x2,y2,conf])
                #vertical stack not append
                detection=np.vstack((detection,currentArray))

                
            


    resultsTracker=tracker.update(detection)
    cv2.line(frame,(line_limit[0],line_limit[1]),(line_limit[2],line_limit[3]),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(frame,(x1,y1,w,h),l=4,rt=5,colorR=(255,0,0))
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2,thickness=1,offset=3)

        #finding coordinates
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)

        #if object is in limit region the count
        if line_limit[0] < cx < line_limit[2] and line_limit[1]-15 < cy < line_limit[1]+15:
            #only append unique Id otherwise count increment many
            #times at the particular regions
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(frame,(line_limit[0],line_limit[1]),(line_limit[2],line_limit[3]),(0,255,0),5)


    # cvzone.putTextRect(frame,f' Count: {len(totalCount)}',(50,50))

    #for graphics
    cv2.putText(frame,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)



    cv2.imshow("Video",frame)
    # cv2.imshow("ImgRegion",imgRegion)

    if cv2.waitKey(1) &  0xFF==ord('q'): 
        break






# cv2.waitKey(0)